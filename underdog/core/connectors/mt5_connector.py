import asyncio
import zmq
import zmq.asyncio
import json
import time
import traceback
import subprocess
import os
import signal
import sys

class MT5Connector:
    def __init__(self, host="127.0.0.1", mt5_exe=None, mql5_script=None,
                 sys_timeout=3.0, heartbeat=5, auto_restart=True):
        self.host = host
        self.mt5_exe = mt5_exe
        self.mql5_script = mql5_script
        self.sys_timeout = sys_timeout
        self.heartbeat = heartbeat
        self.auto_restart = auto_restart

        self.context = zmq.asyncio.Context()
        self.last_data = time.time()
        self.mt5_process = None
        self._init_sockets()

        if self.auto_restart:
            asyncio.create_task(self._watch_mt5())

    # ------------------------------
    # Inicialización y reconexión de sockets
    # ------------------------------
    def _init_sockets(self):
        self.sys_socket = self.context.socket(zmq.REQ)
        self.sys_socket.connect(f"tcp://{self.host}:15555")

        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.connect(f"tcp://{self.host}:15556")

        self.live_socket = self.context.socket(zmq.PULL)
        self.live_socket.connect(f"tcp://{self.host}:15557")

        self.stream_socket = self.context.socket(zmq.PULL)
        self.stream_socket.connect(f"tcp://{self.host}:15558")

    def _reconnect_socket(self, socket, port_name, port, socket_type="REQ"):
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.close()
        except Exception: pass
        socket = self.context.socket(zmq.REQ if socket_type=="REQ" else zmq.PULL)
        socket.connect(f"tcp://{self.host}:{port}")
        print(f"[{port_name} SOCKET] Reconnected")
        return socket

    # ------------------------------
    # SYS request con manejo de OK y backoff
    # ------------------------------
    async def sys_request(self, message: dict):
        attempt = 0
        while True:
            try:
                await self.sys_socket.send_string(json.dumps(message))
                poller = zmq.asyncio.Poller()
                poller.register(self.sys_socket, zmq.POLLIN)
                socks = dict(await poller.poll(timeout=int(self.sys_timeout*1000)))
                if self.sys_socket in socks:
                    resp = await self.sys_socket.recv_string()
                    resp_strip = resp.strip()
                    if not resp_strip:
                        print(f"[SYS REQUEST] Empty response, retry {attempt+1}")
                        raise Exception("Empty SYS response")
                    if resp_strip == "OK":
                        # No JSON, pero no es un error
                        return None
                    try:
                        return json.loads(resp_strip)
                    except json.JSONDecodeError:
                        print(f"[SYS REQUEST] Non-JSON response received: {resp_strip}")
                        raise
                else:
                    print(f"[SYS REQUEST] Timeout retry {attempt+1}")
                    self.sys_socket = self._reconnect_socket(self.sys_socket, "SYS", 15555, "REQ")
            except asyncio.CancelledError:
                raise
            except Exception:
                print(f"[SYS REQUEST] Exception retry {attempt+1}")
                traceback.print_exc()
                self.sys_socket = self._reconnect_socket(self.sys_socket, "SYS", 15555, "REQ")
            attempt += 1
            await asyncio.sleep(min(0.5*(2**attempt), 5))

    # ------------------------------
    # Listeners asíncronos con heartbeat
    # ------------------------------
    async def listen_socket(self, socket, callback, port_name, port):
        while True:
            try:
                msg = await asyncio.wait_for(socket.recv_string(), timeout=self.heartbeat)
                self.last_data = time.time()
                callback(json.loads(msg))
            except asyncio.TimeoutError:
                print(f"[{port_name} SOCKET] No data for {self.heartbeat}s, reconnecting...")
                socket = self._reconnect_socket(socket, port_name, port)
            except Exception:
                print(f"[{port_name} SOCKET] Exception:")
                traceback.print_exc()
                socket = self._reconnect_socket(socket, port_name, port)

    async def listen_data(self, callback): return await self.listen_socket(self.data_socket, callback, "DATA", 15556)
    async def listen_live(self, callback): return await self.listen_socket(self.live_socket, callback, "LIVE", 15557)
    async def listen_stream(self, callback): return await self.listen_socket(self.stream_socket, callback, "STREAM", 15558)
    async def run_listeners(self, data_cb, live_cb, stream_cb):
        await asyncio.gather(self.listen_data(data_cb), self.listen_live(live_cb), self.listen_stream(stream_cb))

    # ------------------------------
    # Operaciones de trading
    # ------------------------------
    async def send_order(self, actionType, symbol, volume, price=None, sl=0, tp=0, deviation=5, comment="python-bot", order_id=0):
        msg = {
            "action": "TRADE",
            "actionType": actionType,
            "symbol": symbol,
            "volume": volume,
            "price": price if price else 0,
            "stoploss": sl,
            "takeprofit": tp,
            "deviation": deviation,
            "comment": comment,
            "id": order_id
        }
        return await self.sys_request(msg)

    async def close_position(self, position_id):
        return await self.send_order("POSITION_CLOSE_ID", "", 0, order_id=position_id)

    async def modify_position(self, position_id, sl=None, tp=None):
        return await self.send_order("POSITION_MODIFY", "", 0, sl=sl or 0, tp=tp or 0, order_id=position_id)

    async def get_account_info(self): return await self.sys_request({"action": "ACCOUNT"})

    async def request_history(self, symbol, chartTF, fromDate, toDate=None):
        msg = {"action": "HISTORY", "actionType": "DATA", "symbol": symbol, "chartTF": chartTF, "fromDate": fromDate}
        if toDate: msg["toDate"] = toDate
        return await self.sys_request(msg)

    # ------------------------------
    # Watchdog MT5 y relaunch automático
    # ------------------------------
    async def _watch_mt5(self):
        while True:
            await asyncio.sleep(3)
            if self.last_data + self.heartbeat*2 < time.time():
                print("[WATCHDOG] No data detected, checking MT5...")
                if not self._is_mt5_alive():
                    print("[WATCHDOG] MT5 not running, relaunching...")
                    self._launch_mt5()

    def _is_mt5_alive(self):
        if self.mt5_process is None:
            return False
        return self.mt5_process.poll() is None

    def _launch_mt5(self):
        if self.mt5_exe is None:
            print("[WATCHDOG] MT5 exe path not provided, cannot launch")
            return
        try:
            if self.mql5_script:
                self.mt5_process = subprocess.Popen([self.mt5_exe, "/portable", "/script:"+self.mql5_script],
                                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self.mt5_process = subprocess.Popen([self.mt5_exe, "/portable"],
                                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[WATCHDOG] MT5 launched")
        except Exception:
            print("[WATCHDOG] Failed to launch MT5")
            traceback.print_exc()
