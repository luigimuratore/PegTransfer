# Peg transfer — risultati e prossime modifiche

## 2 ottobre 2026 — primo checkpoint completo

**Risultato: 100 trasferimenti riusciti su 100 prove in Isaac Sim**, sempre con il peg inizialmente su L5 e la posa finale su R2. Nessun fallimento o timeout.

PSM1 raggiunge il peg, lo prende, lo solleva e lo porta verso PSM2. PSM2 lo prende; PSM1 apre la pinza e si allontana. PSM2 trasporta il peg su R2, lo posa, apre la pinza e si allontana.

### Come ha imparato

Prima abbiamo fatto completare il task a un controller programmato e registrato **6 dimostrazioni riuscite**. Poi abbiamo addestrato la rete a imitare le azioni di quelle dimostrazioni. Questo metodo si chiama **Behavior Cloning (BC)**.

La rete comanda il movimento nelle tre direzioni e l'apertura/chiusura della pinza del braccio che sta lavorando. È la stessa rete per entrambi i PSM.

**SAC è già implementato per il successivo apprendimento tramite reward, ma questo checkpoint è stato addestrato solo per imitazione.** Non abbiamo ancora eseguito training SAC tramite tentativi e ricompense. Anche il curriculum, cioè imparare gradualmente le parti del task, è disponibile ma non è stato necessario per ottenere questo primo risultato.

Parametri principali:

| Parametro | Valore |
|---|---|
| Dimostrazioni | 6 task completi, 8040 passi registrati |
| Aggiornamenti della rete | 6000 |
| Rete | 2 strati da 256 neuroni |
| Learning rate — quanto cambia la rete a ogni aggiornamento | 0,0001 |
| Batch — esempi usati per ogni aggiornamento | 256 |

### Cosa è autonomo e cosa è assistito

Durante le 100 prove, **la rete ha scelto movimento e comando della pinza senza ricevere le azioni del controller esperto**.

Il codice continua a scegliere l'ordine delle fasi e i punti da raggiungere, e mantiene fermo il braccio che non sta lavorando. La presa è assistita: quando viene riconosciuta, il codice fa seguire il peg alla pinza. Non abbiamo ancora dimostrato una presa mantenuta soltanto da forza e attrito.

Il risultato vale per **L5 e R2 fissi**. Nuove posizioni e perturbazioni devono ancora essere provate.

### Breve timeline

| Passaggio | Risultato |
|---|---|
| Prime reti per la sola presa e il sollevamento | 0/100 successi |
| Primo controller completo | PSM1 riusciva; PSM2 faceva perdere il peg |
| Correzione dell'approccio PSM2 dal bordo opposto | Controller: 3/3 trasferimenti completi |
| Raccolta delle dimostrazioni | 6/6 trasferimenti completi |
| Rete addestrata per imitazione | **100/100 trasferimenti completi** |

### File di riferimento

Checkpoint da conservare:

`logs/sac_sequence/2026-10-02_15-11-19_sequence_bc/final.zip`

Report della valutazione:

`logs/sac_sequence/eval_full_unguided_2026-10-02_15-11-38.json`

**Prossima modifica: da decidere insieme.** Aggiungeremo qui il cambiamento e i nuovi risultati per confrontarli con questo checkpoint. I dettagli tecnici restano in [PEG_TRANSFER.md](PEG_TRANSFER.md) e nel manifest del checkpoint.

5 ottobre 2026: questa versione è conservata come **checkpoint1**, con codice e modello originali. `bash peg_checkpoint1.sh play` riproduce il task salvato anche dopo modifiche al progetto; `bash peg_checkpoint1.sh video` registra il video via SSH. Rimane necessario l'ambiente Isaac originale. Questo salvataggio non è un nuovo training o una nuova valutazione.
