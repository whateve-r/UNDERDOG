import asyncio
import json
import zmq
import zmq.asyncio
import sys

# --- 🔹 Evitar warning de Proactor loop en Windows ---
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class MT5Connector:
    def __init__(self, host="127.0.0.1", sys_port=15555, data_port=15556, live_port=15557, str_port=15558, debug=True):
        self.host = host
        self.sys_port = sys_port
        self.data_port = data_port
        self.live_port = live_port
        self.str_port = str_port
        self.debug = debug  

        self.context = zmq.asyncio.Context()

        self.sys_socket = self.context.socket(zmq.REQ)
        self.sys_socket.connect(f"tcp://{self.host}:{self.sys_port}")

        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.connect(f"tcp://{self.host}:{self.data_port}")

        self.live_socket = self.context.socket(zmq.PULL)
        self.live_socket.connect(f"tcp://{self.host}:{self.live_port}")

        self.stream_socket = self.context.socket(zmq.PULL)
        self.stream_socket.connect(f"tcp://{self.host}:{self.str_port}")

    async def sys_request(self, request: dict):
        """Send a request to SYS socket and wait for JSON response safely."""
        attempt = 0
        while attempt < 5:
            try:
                await self.sys_socket.send_json(request)
                msg = await self.sys_socket.recv_string()
                if not msg.strip():
                    if self.debug:
                        print("[SYS REQUEST] Empty message, retrying...")
                    attempt += 1
                    await asyncio.sleep(min(0.5*(2**attempt), 5))
                    continue
                try:
                    return json.loads(msg)
                except json.JSONDecodeError:
                    if self.debug:
                        print("[SYS REQUEST] Non-JSON response received:", msg)
                    attempt += 1
                    await asyncio.sleep(min(0.5*(2**attempt), 5))
            except asyncio.CancelledError:
                raise
        raise Exception("SYS request failed after 5 retries")

    async def get_account_info(self):
        return await self.sys_request({"action": "ACCOUNT"})

    async def get_balance_info(self):
        return await self.sys_request({"action": "BALANCE"})

    async def request_history(self, symbol, timeframe, from_date, to_date=None):
        req = {
            "action": "HISTORY",
            "actionType": "DATA",
            "symbol": symbol,
            "chartTF": timeframe,
            "fromDate": int(from_date),
        }
        if to_date:
            req["toDate"] = int(to_date)
        return await self.sys_request(req)

    async def run_data_listener(self, callback):
        while True:
            try:
                msg = await self.data_socket.recv_string()
                callback(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print("[DATA LISTENER ERROR]", e)
                await asyncio.sleep(1)

    async def run_live_listener(self, callback):
        while True:
            try:
                msg = await self.live_socket.recv_string()
                callback(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print("[LIVE LISTENER ERROR]", e)
                await asyncio.sleep(1)

    async def run_stream_listener(self, callback):
        while True:
            try:
                msg = await self.stream_socket.recv_string()
                callback(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print("[STREAM LISTENER ERROR]", e)
                await asyncio.sleep(1)


async def main():
    mt5 = MT5Connector(debug=True)

    # 🔹 DATA callback
    def data_cb(msg):
        try:
            data = json.loads(msg)
            print("[DATA]", json.dumps(data, indent=2))
        except:
            print("[DATA RAW]", msg)

    # 🔹 LIVE callback
    def live_cb(msg):
        try:
            data = json.loads(msg)
            print("[LIVE]", json.dumps(data, indent=2))
        except:
            print("[LIVE RAW]", msg)

    # 🔹 STREAM callback
    def stream_cb(msg):
        try:
            data = json.loads(msg)
            print("[STREAM]", json.dumps(data, indent=2))
        except:
            print("[STREAM RAW]", msg)

    # 🔹 Start listeners
    tasks = [
        asyncio.create_task(mt5.run_data_listener(data_cb)),
        asyncio.create_task(mt5.run_live_listener(live_cb)),
        asyncio.create_task(mt5.run_stream_listener(stream_cb)),
    ]

    try:
        # 🔹 Test SYS requests
        account = await mt5.get_account_info()
        print("[ACCOUNT]", json.dumps(account, indent=2))

        balance = await mt5.get_balance_info()
        print("[BALANCE]", json.dumps(balance, indent=2))

        # 🔹 Optional: request historical data
        # import time
        # hist = await mt5.request_history("EURUSD", "M1", int(time.time())-3600)
        # print("[HISTORY]", json.dumps(hist, indent=2))

        # Keep running to receive DATA, LIVE, STREAM
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("Exiting...")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
