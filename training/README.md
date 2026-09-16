# training — o lado Python

## Ambiente

Criado com o **único Python arm64 desta máquina**:

```bash
/usr/bin/python3 -m venv training/.venv     # arm64, 3.9.6 (Xcode CLT)
training/.venv/bin/python -m pip install -r training/requirements.txt
```

Verificado em 2026-09-16: `tensorflow 2.20.0`, `numpy 2.0.2`, `arch arm64`.

### Por que não o `python3` do PATH

| Interpretador | Arch | Versão | TensorFlow |
|---|---|---|---|
| `/usr/local/bin/python3` (Homebrew) | **x86_64** (Rosetta) | 3.13.5 | ❌ sem wheels |
| `/usr/bin/python3` (sistema) | **arm64** | 3.9.6 | ✅ 2.20.0 |

O Homebrew desta máquina está instalado no prefixo Intel (`/usr/local`), não no
`/opt/homebrew`. Tudo que vem dele é x86_64 emulado — inclusive o Python. Para ML isso é
inviável: além de não haver wheels, o desempenho cai muito.

### Plano de saída (quando o 3.9 incomodar)

O 3.9.6 está em fim de vida e prende as versões de tudo. Duas saídas, em ordem de
preferência:

1. **Instalar o Python oficial arm64** de python.org (3.12 ou 3.13) — não conflita com
   nada e é o caminho mais curto. Depois: recriar o venv apontando para ele.
2. **Reinstalar o Homebrew em `/opt/homebrew`** (o prefixo arm64), mantendo ou não o de
   `/usr/local`. Resolve o problema na raiz, mas mexe em tudo que você já instalou.

Não é urgente: o TensorFlow 2.20 no 3.9.6 cobre as quatro etapas do roteiro.

## Layout

```
src/     código: dados, baselines, treino, export, verificação
tests/   pytest — formato Given/When/Then (Regra 02)
data/    dataset baixado (ignorado pelo git)
```

`src/` cresce um passo por vez, na ordem da spec 0001. O que já existe está no `CLAUDE.md`,
seção "Estado atual".
