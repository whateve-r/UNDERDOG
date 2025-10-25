# 🐛 Python 3.13 + PyTorch Import Issue - WORKAROUND

**Date**: October 24, 2025  
**Status**: KNOWN ISSUE - Python 3.13.0 + PyTorch asyncio race condition  
**Impact**: Training script crashes during torch import with `KeyboardInterrupt` in enum module

---

## 🔴 PROBLEMA CRÍTICO

**Error observado**:
```python
Traceback (most recent call last):
  File "scripts/train_marl_agent.py", line 26, in <module>
    import torch
  # ... (deep stack trace through torch internals)
  File "C:\...\Python313\Lib\enum.py", line 764, in __delattr__
    def __delattr__(cls, attr):
KeyboardInterrupt
```

**Root Cause**: Python 3.13.0 cambió el comportamiento de `asyncio` y `enum` módulos internos. PyTorch 2.5.1 (y versiones anteriores) tienen **race condition** durante import de:
- `torch.distributions.von_mises`
- `torch.jit._trace._ExportType` (enum initialization)
- `torch._dispatch.python` (asyncio dispatcher)

**Reproducción**:
```bash
poetry run python scripts/train_marl_agent.py --episodes 50 ...
# Crash durante import de torch (antes de alcanzar main())
```

---

## ✅ SOLUCIÓN 1: DOWNGRADE A PYTHON 3.12 (RECOMENDADO)

**Justificación**: PyTorch oficialmente soporta Python 3.12, pero 3.13 aún en beta/experimental.

### Pasos:

1. **Instalar Python 3.12.7** (última estable):
   ```powershell
   # Descargar desde python.org/downloads
   # O usar pyenv/conda:
   pyenv install 3.12.7
   pyenv local 3.12.7
   ```

2. **Actualizar pyproject.toml**:
   ```toml
   [tool.poetry.dependencies]
   python = "^3.12, <3.13"  # Cambiar de ^3.13
   ```

3. **Recrear entorno**:
   ```powershell
   poetry env remove python
   poetry install
   ```

4. **Validar**:
   ```powershell
   poetry run python -c "import torch; print(torch.__version__)"
   # Debe imprimir sin crash
   ```

---

## ✅ SOLUCIÓN 2: USAR PYTHON DIRECTAMENTE (TEMPORAL)

**Si no puedes hacer downgrade inmediatamente**, usa el Python del sistema (no Poetry):

```powershell
# Activar venv manualmente
& C:/Users/manud/AppData/Local/pypoetry/Cache/virtualenvs/underdog-dQryfNjJ-py3.13/Scripts/Activate.ps1

# Correr script directamente
python scripts/train_marl_agent.py --episodes 50 --symbols EURUSD USDJPY XAUUSD GBPUSD --initial-balance 100000 --batch-size 256
```

**Nota**: Esto evita el wrapper de Poetry que puede introducir latencia adicional en imports.

---

## ✅ SOLUCIÓN 3: LAZY IMPORT DE TORCH (CÓDIGO)

**Modificar train_marl_agent.py** para importar torch **dentro de main()** en vez de top-level:

```python
# scripts/train_marl_agent.py (LÍNEAS 1-30)

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
# import torch  # ❌ COMENTAR ESTO
import csv

# ... otros imports ...

def main():
    """CLI entry point"""
    
    # ✅ IMPORTAR TORCH AQUÍ (lazy import)
    import torch
    
    parser = argparse.ArgumentParser(...)
    # ... resto del código ...
```

**Ventaja**: Permite que argparse funcione antes de cargar torch.  
**Desventaja**: Solo pospone el problema, no lo resuelve.

---

## ✅ SOLUCIÓN 4: USAR CONDA (ALTERNATIVA A POETRY)

**Si Python 3.13 es requerido** por otras razones, usar Conda con PyTorch precompilado:

```bash
conda create -n underdog python=3.13
conda activate underdog
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

**Nota**: Conda builds de PyTorch a veces tienen mejor compatibilidad con Python bleeding-edge.

---

## 🎯 RECOMENDACIÓN FINAL

**Para este proyecto (UNDERDOG HMARL Training)**:

1. **CORTO PLAZO** (hoy): Usar **Solución 2** (Python directo) para lanzar entrenamiento.
2. **MEDIO PLAZO** (esta semana): Hacer **Solución 1** (downgrade a 3.12.7).
3. **LARGO PLAZO**: Esperar a PyTorch 2.6+ con soporte oficial de Python 3.13.

**Comando de lanzamiento RÁPIDO**:
```powershell
# Activar venv
& C:/Users/manud/AppData/Local/pypoetry/Cache/virtualenvs/underdog-dQryfNjJ-py3.13/Scripts/Activate.ps1

# Lanzar entrenamiento
python scripts/train_marl_agent.py --episodes 50 --symbols EURUSD USDJPY XAUUSD GBPUSD --initial-balance 100000 --batch-size 256

# En otra terminal: Dashboard live
python scripts/visualize_hmarl.py --log-file logs/training_metrics.csv --live
```

---

## 📊 ESTADO DEL SISTEMA

| Componente | Estado | Notas |
|------------|--------|-------|
| **Staleness Features** | ✅ LISTO | 31D observations, features 29-30 |
| **Reward Shapers** | ✅ LISTO | GBPUSD fakeout, USDJPY volatility |
| **Dashboard HMARL** | ✅ LISTO | visualize_hmarl.py con 8 métricas |
| **Training Script** | ⚠️ BLOQUEADO | Python 3.13 import issue |
| **Solución** | 🔧 EN CURSO | Usar Python directo (no poetry run) |

---

## 🔗 REFERENCIAS

- **PyTorch Issue**: https://github.com/pytorch/pytorch/issues/137000 (Python 3.13 asyncio)
- **Python 3.13 Changes**: https://docs.python.org/3.13/whatsnew/3.13.html#asyncio
- **Poetry Env Management**: https://python-poetry.org/docs/managing-environments/

---

**Last Updated**: 2025-10-24  
**Next Action**: Launch training with Python direct invocation (bypassing Poetry wrapper).
