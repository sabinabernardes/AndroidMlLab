# Registro de atrito

Uma entrada por momento em que o método ou o agente falhou. Serve para virar regra —
que é o que impede o mesmo erro de acontecer três vezes.

---

## Montagem do repositório (2026-09-16)

**Duas armadilhas de ambiente, nenhuma delas de ML**

1. **Máquina arm64, Homebrew em `/usr/local`.** O `python3` do PATH é x86_64 sob Rosetta
   e não tem wheels de TensorFlow. Só o `/usr/bin/python3` (3.9.6) é nativo. Descoberto
   *antes* de escrever código, porque a primeira coisa feita foi checar o ambiente — se
   tivesse sido depois, o sintoma seria um `pip install` falhando sem explicação óbvia.
   → Registrado em `CLAUDE.md` e em `training/README.md`, com plano de saída.

2. **`ai-edge-litert`: wheels quebrados nas versões novas.** As 1.4.0 e 2.0.x instalam e
   falham no import em macOS/arm64 (`libpywrap_litert_common.dylib` ausente). A 1.3.0
   funciona. E a alternativa óbvia — `tf.lite.Interpreter` — está **deprecada e marcada
   para remoção pelo próprio TensorFlow 2.20**, que avisa isso num warning que passaria
   batido.
   → Versão fixada com o motivo escrito no `requirements.txt`. Um `==` sem comentário é
   uma decisão que ninguém vai conseguir revisitar depois.

**O erro do agente nesta sessão: spec calibrada para a pessoa errada**

A primeira versão da `0001` terminava com três decisões "suas": o piso de acurácia, quais
critérios de aceite cortar, e se valia medir a regressão logística. **As três exigem ter
treinado um modelo antes.** Pedir isso a quem está começando não é respeitar a autonomia
da pessoa — é transferir uma decisão que ela não tem como tomar e chamar o chute dela de
aprovação.

Só apareceu porque a informação "não sei nada de ML" foi dita explicitamente. O agente
tinha pistas para inferir (a escolha do próprio repositório era "quero aprender ML") e
não inferiu.

→ *Virou regra:* a seção **"Quem trabalha aqui"** no `CLAUDE.md`, que manda decidir e
justificar em vez de perguntar, quando a pergunta exige experiência que a pessoa não
tem. E `docs/CONCEITOS.md`, que é a densidade da spec movida para um lugar onde ela pode
ser consultada em vez de presumida.

**O que o método poupou**

O `pytest` de ambiente (3 testes, nenhuma linha de treino) pegou as duas armadilhas
acima antes de existir qualquer código de feature. Se o primeiro comando do repo tivesse
sido "treinar um modelo", os dois erros de ambiente apareceriam misturados com erros de
ML — e nenhum dos dois seria óbvio.
