# Peg transfer con due dVRK PSM — sequenza SAC v1

## Versione attuale e prove disponibili

Dal 2 ottobre 2026 il percorso consigliato è **`Isaac-Peg-Transfer-Dual-PSM-Sequence-v1`**, separato dalla precedente pipeline. **Spawn L5 fisso, destinazione R2 fissa, nessuna randomizzazione delle sedi.** Il curriculum riguarda le abilità di entrambi i PSM e l'intera sequenza. Un'eventuale estensione ad altre sedi viene dopo una valutazione reale del trasferimento completo.

Le prove utente v3/v4/v5 della vecchia rete SAC hanno tutte dato zero grasp e lift su 100 episodi. Il controller esperto ha invece dimostrato approach/grasp/lift e tre recuperi reali; questo non dimostra handover o posa. Anche mascherando le azioni inutilizzate, v5 non si avvicina più del reset (88,66 mm). Ripetere quei training non ha prodotto progressi. Il nuovo codice cambia rappresentazione delle azioni, osservazioni dell'attore, obiettivi, reward, imitazione e curriculum.

**Nessuna simulazione Isaac né training lungo è stata avviata dall'agente. Nessun checkpoint sequence-v1 è ancora addestrato o validato in Isaac Sim.** I 22 test CPU passano; includono un surrogato cinematico ideale che percorre le venti tappe usando la logica di presa/collisione. Quel test verifica la logica, non raggiungibilità, contatti PhysX o prestazioni della policy. Le metriche della nuova policy sono quindi **non misurate**, non 100%.

Versioni locali controllate: Isaac Sim 5.1.0.0, Isaac Lab 0.48.0, PyTorch 2.7.0+cu128, stable-baselines3 2.9.0, Gymnasium 1.2.0. SAC usa le API SB3 locali; il training rifiuta versioni SB3 diverse finché non vengono ricontrollate. Il wrapper imposta il namespace Python `simulation`, risolvendo l'import mancante osservato via SSH.

### Primo probe reale e correzione della tenuta assistita

Esecuzione utente `logs/sac_sequence/baseline_full_2026-10-02_14-25-13.json`, fingerprint `8597f131d7df315b6b0ed75f6df7ec628d50af0486f37ae9f6ceb22c965f4a03`: **3/3 grasp PSM1, lift stabile e zona transfer; 0/3 grasp PSM2, handover, posa e successo completo**. Tutti gli episodi finiscono in `receiver_reach`, dopo 75 controlli di quella tappa, per `attachment_lost`; zero timeout, 23 movimenti assistiti bloccati. È una baseline analitica, non una valutazione della rete.

La prima traccia mostra root peg fermo a z≈64,859 mm fino al passo 771, poi 64,194 mm al passo 772 e 60,160 mm al passo 773. PSM1 resta fermo con pinza a 0,07 rad; PSM2 è ancora a circa 8 mm dal grasp prima del disturbo. Il terminale misura root z≈52,201 mm e quaternion-vector norm≈0,00695. La traccia originale non registra quaternion/possessore per substep: non permette di attribuire con certezza il primo impulso a uno specifico collider.

Nel codice precedente una rotazione oltre 0,005 interrompeva il servo di un oggetto già tenuto, senza ripristinare la posa né annullare la velocità. Un piccolo disturbo poteva quindi lasciare gravità/velocità attive, accumulare caduta e superare lo stretch di 8 mm. Questo meccanismo è riprodotto da una regressione CPU ed è compatibile con i dati reali; la conferma della correzione richiede il nuovo probe PhysX.

La nuova `peg_transfer_sequence/assistance.py` conserva l'ultima posa upright sicura mentre il peg è detenuto. Corregge piccoli disturbi dopo il substep e azzera la velocità come previsto dal modello assistito. Una traslazione richiesta e rifiutata dal controllo board/post mantiene quella posa, invece di lasciare cadere il peg. Conserva lo stretch di 8 mm e la soglia iniziale di capture di 0,005; disturbi di traslazione oltre 4 mm, quaternion-vector norm oltre 0,04 o ripristino non sicuro causano rilascio. Nessun collider viene disabilitato. Si tratta di proiezione assistita della posa, non di stabilità dimostrata di una presa a forze: gli impulsi piccoli sull'oggetto trattenuto vengono compensati dal servo.

La traccia nuova registra anche quaternion, holder/latch, conteggio delle correzioni, picchi dei disturbi PhysX prima della correzione e motivo di rottura (`tool_stretch`, `large_rotation`, `large_displacement`, `unsafe_restoration`). Il terminale conserva questi dati prima del reset. I sorgenti legacy restano invariati; report e checkpoint precedenti sono conservati. La versione corretta ha un nuovo fingerprint e **deve superare nuovamente la baseline completa prima della raccolta/training**.

### Secondo probe reale: ingresso laterale PSM2 e nuovo percorso

Il probe video utente `baseline_full_2026-10-02_14-44-51.json`, fingerprint `1629df3a05614492c0858938fc68dfc1719ce5bd71695e32bf772535e0873af8`, completa tutti e tre gli episodi ma fallisce ancora in `receiver_reach`: 3/3 grasp/lift/transfer PSM1, zero ricezioni e successi completi. La tenuta ora corregge i primi piccoli disturbi, ma il terminale identifica `large_rotation`: quaternion-vector norm prima del rilascio circa 0,081–0,097, ossia 9,2–11,1 gradi. Non è un timeout e non è un percorso bloccato dai post: `blocked_moves=0`. La pinza PSM2 resta aperta a circa 0,5 rad. I primi disturbi iniziano quando il tip è ancora circa 8 mm dal grasp; 37–40 correzioni precedono il rilascio.

Sono stati esaminati il video MP4 locale e lo USD PSM nella cache Omniverse, leggendo USD senza avviare Isaac. I collider delle dita sono mesh con `physics:collisionEnabled=True`; le dita aperte hanno estensione prima del punto geometrico di capture. La crescita della perturbazione lungo l'ingresso da +X è coerente con una spinta della pinza aperta sul lato del peg. Il report non identifica la coppia esatta di collider tramite un sensore di contatto, quindi questa attribuzione rimane un'inferenza da video, geometria e traccia; non si aumenta la soglia di rottura per nasconderla.

La nuova `peg_transfer_sequence/geometry.py` calibra **solo PSM2** sul bordo +Y: offset dal root peg `(0.014, 0.009, -0.006)` m. PSM1 conserva il punto originale sul bordo -Y. PSM2 passa sopra e all'esterno del bordo +Y (15 mm dal nuovo grasp), scende fuori dalla sagoma e arriva aperto a un offset +Y di 3,5 mm. Non scorre più lungo +X verso il vecchio punto interno al lato destro. Il tip finale è oltre il bordo Y del collider e dentro il raggio di capture di 4 mm; il test CPU verifica questa relazione, mentre la clearance delle dita nella loro orientazione reale va ancora validata in PhysX.

Presa/latch, coordinate di approach e metriche di distanza usano lo stesso nuovo punto. Nessuna soglia di capture, rottura o successo viene allargata; collider e servo restano quelli della revisione precedente. La nuova traccia aggiunge anche quaternion dei tool e posizioni dei giunti, per poter controllare le dita se il problema persiste. Report, demo e manifest dichiarano `receiver_approach=positive_y_edge`. La CLI stampa cambio di tappa e avanzamento ogni 100 controlli: un terminale silenzioso dopo il setup non va interpretato automaticamente come un blocco.

Rilanciare `bash peg_sequence.sh tests` e poi `bash peg_sequence.sh probe` oppure `bash peg_sequence.sh probe --video`. Non raccogliere demo o avviare SAC finché il nuovo trasferimento completo non passa 3/3. Il probe headless precedente interrotto con Ctrl-C non è una valutazione completa. Le warning GLFW in SSH non hanno impedito al probe video del 14:44:51 di terminare e scrivere il video; non dimostrano la causa del fallimento robotico.

### Baseline reale L5 → R2 superata

Il report inviato dall'utente `logs/sac_sequence/baseline_full_2026-10-02_15-00-34.json`, fingerprint `b827e9291195ffc991d8bb8136edf98c2ccfc30699ca4d2244c59f9437d6313a`, verifica **3/3 trasferimenti completi**: grasp PSM1, lift stabile, zona transfer, reach/grasp PSM2, handover e posa su R2. Zero timeout, fallimenti, movimenti bloccati e correzioni di contatto riportate. Alla fine il peg è rilasciato (`holder=0`), entrambi i jaw sono aperti e i tool sono lontani dalla presa. La prova preliminare `baseline_full_2026-10-02_14-58-34.json` aveva già completato 1/1 episodio. La baseline a tre episodi supera il gate operativo per raccogliere dimostrazioni; non è una stima robusta di generalizzazione (Wilson 95% su 3/3: 0,439–1,000).

Questa evidenza riguarda il controller analitico con `guidance=1` e presa assistita, con spawn L5 e target R2 fissi. `checkpoint=null`, `learned_active_motor_without_guidance=false` e `physical_grasp_verified=false`: nessuna policy appresa completa o presa basata sulle sole forze è ancora dimostrata. Proseguire con `bash peg_sequence.sh collect`; solo dopo una raccolta completa riuscita, `bash peg_sequence.sh bc` e `bash peg_sequence.sh eval`. La selezione automatica usa gli ultimi artefatti compatibili. Il curriculum SAC viene deciso sulla valutazione della rete a guida zero.

### Policy imitativa completa: valutazione reale a guida zero

Report sintetico della milestone e registro delle revisioni: [PEG_TRANSFER_TIMELINE.md](PEG_TRANSFER_TIMELINE.md).

Il report utente `logs/sac_sequence/eval_full_unguided_2026-10-02_15-11-38.json` valuta il checkpoint `logs/sac_sequence/2026-10-02_15-11-19_sequence_bc/final.zip` sullo stesso fingerprint `b827e9291195ffc991d8bb8136edf98c2ccfc30699ca4d2244c59f9437d6313a`. Esito: **100/100 grasp PSM1, lift stabile, grasp PSM2, handover, posa e trasferimenti completi**, zero timeout, fallimenti e movimenti bloccati; `complete=true`, seed 123, 16 ambienti, `guidance=[0.0]`, baseline disabilitata. Il manifest locale conferma `imitation_only=true`, 6000 aggiornamenti BC e 8040 transizioni dimostrative: è l'attore SAC addestrato per imitazione, senza ancora fine tuning RL. Non è un risultato di SAC online.

La guida motoria analitica del braccio attivo è disabilitata; restano sequenziamento e controllo del braccio inattivo programmati, oltre alla presa assistita. Spawn L5 e target R2 sono fissi; le traiettorie parallele sono quasi identiche. I 100 successi confermano ripetibilità in questa configurazione, non robustezza a spawn, orientazioni o perturbazioni diverse. Prima di avviare ulteriori aggiornamenti RL conservare questo checkpoint e verificare anche il video/GUI e un'altra valutazione con seed 999.

Per vedere esattamente questa policy, dal terminale nel desktop grafico della workstation, dopo `conda activate PegTransfer` e dalla root del repository:

```bash
bash peg_sequence.sh play logs/sac_sequence/2026-10-02_15-11-19_sequence_bc/final.zip --episodes 3
```

Il comando apre Isaac Sim con un ambiente, task completo e guida zero. Su SSH senza desktop usare invece `bash peg_sequence.sh video logs/sac_sequence/2026-10-02_15-11-19_sequence_bc/final.zip`; salva la registrazione in `logs/sac_sequence/sequence_videos/<nome_report>/`. Non imposta artificialmente `DISPLAY`: per vedere la GUI da remoto occorre una sessione desktop della workstation.

## Design della sequenza e autonomia effettiva

Il controllore gerarchico sceglie una tappa tramite condizioni misurate. SAC comanda **tre traslazioni e una pinza continue del braccio attivo**. I comandi sono trasformati nei quattordici comandi IK/pinza effettivi dei due PSM. I comandi di rotazione sono nulli; feedback di tenuta del braccio inattivo e scelta della tappa restano programmati. Le pinze usano [-1,+1]: -1 richiede 0,07 rad, +1 richiede 0,5 rad; la logica controlla gli angoli reali.

`guidance=1` applica il controllore analitico con piccoli residui appresi; `guidance=0` elimina la componente esperta **dalla traslazione e dalla pinza del braccio attivo**. Anche a zero la scelta delle tappe e il controllo del braccio inattivo restano analitici. È una strategia ibrida autonoma, non una policy monolitica che apprende anche sequenziamento e rotazioni. Nei report queste semplificazioni sono dichiarate. Il successo della baseline analitica non viene etichettato come successo della rete.

| Skill del curriculum | Sequenza verificata prima della terminazione |
|---|---|
| `approach` | Assestamento → PSM1 sopra e all'esterno del peg → arrivo aperto nel raggio di capture |
| `grasp` | Approach → chiusura PSM1 sul posto → aggancio effettivo |
| `lift` | Grasp → estrazione verticale dal post → lift stabile |
| `transfer` | Lift → trasporto PSM1 alla zona di handover |
| `receive` | Transfer → PSM2 sopra/all'esterno → arrivo aperto → chiusura e latch effettivo |
| `handover` | Receive → apertura PSM1 → peg detenuto da PSM2 → ritirata PSM1 |
| `full` | Handover → trasporto PSM2 → allineamento foro/R2 → inserimento → apertura PSM2 → ritirata → posa stabile |

Le venti tappe sono `settle`, `donor_above`, `donor_outside`, `donor_reach`, `donor_close`, `donor_lift`, `donor_hold`, `donor_transfer`, `receiver_above`, `receiver_outside`, `receiver_reach`, `receiver_close`, `donor_release`, `donor_retreat`, `receiver_transport`, `receiver_align`, `receiver_insert`, `receiver_release`, `receiver_retreat`, `placement_settle`. Il completamento è la tappa 20. Ogni episodio riparte dall'inizio reale: nessun reset con peg già agganciato o direttamente nelle mani di PSM2.

PSM1/PSM2 arrivano aperti prima di passare alla chiusura. Le tappe di movimento richiedono cinque campioni consecutivi entro la tolleranza; le prese richiedono il dwell fisico esistente. La sola distanza o il passare del tempo non prova presa, handover o posa. La posa finale richiede anche evidenza fresca di assestamento: una metrica storica `full_success=True` non consente di ignorare una posa successivamente disturbata.

## Osservazioni, reward e apprendimento

Il critic riceve 112 valori: errori dei goal, peg e tool, stato/velocità dei giunti, pinze, possessore, latch, evidenze fisiche, contatori e tappa. L'attore riceve 29 valori selezionati: errore del goal nel frame del braccio, angoli delle due pinze, tappa, braccio attivo e livello di guida. Non riceve tempo dell'episodio, reward precedente, azione esperta o giunti da usare come scorciatoia temporale. La stessa rete serve entrambi i bracci.

Reward per controllo a 50 Hz:

```text
Phi(s) = 2 * tappa + exp(-errore_goal / 0.03) * braccio_attivo
r = 0.999 * Phi(s_next) - Phi(s)
    + eventi_fisici_once([4, 12, 6, 12, 10, 50])
    + 2 * avanzamento_tappa
    + 20 * successo_skill - 20 * fallimento
    - 0.005 - 0.002 * somma(azione_policy^2)
    - 0.2 * nuovi_movimenti_peg_bloccati
```

Gli eventi sono grasp PSM1, lift stabile, grasp PSM2, handover, posa e successo completo; vengono pagati una sola volta. Non si guadagna un bonus positivo ripetuto restando vicino al peg o chiudendo/riaprendo. La funzione potenziale viene azzerata alle vere terminazioni, conservata ai timeout per il bootstrap. Un movimento verso R2 senza tutte le tappe precedenti non conclude il task. Fallimenti: perdita della presa durante il trasporto, presa prematura, uscita dai limiti del peg/tool e budget della tappa esaurito. I report elencano le cause.

Episodio 60 s = 3.000 controlli; budget locale 600 controlli = 12 s per tappa. Un budget maggiore è disponibile con `--stage_budget 900 --episode_seconds 90`, ma richiede demo raccolte con lo stesso orizzonte. I reset asincroni conservano lo stato delle altre istanze. Il replay salva l'osservazione terminale **prima** del reset e distingue timeout da vera terminazione.

La raccolta produce traiettorie complete a guida 1, 0,5 e 0 con piccola perturbazione delle azioni. Ogni transizione conserva l'azione realmente eseguita per SAC e, separatamente, l'etichetta esperta dello stesso stato per BC. Nessuna transizione fisica viene inventata; l'archivio viene salvato solo se **tutti** i trasferimenti richiesti riescono. Il campionamento delle demo bilancia tappa e guida, evitando che il lungo approach domini chiusura e handover.

L'attore usa un unico aggiornamento con obiettivo `loss_SAC + bc_weight * loss_BC`, invece di due aggiornamenti Adam in conflitto. BC pesa le traslazioni quattro volte la pinza, limita lo std dell'attore e mantiene un'ancora dimostrativa durante RL. Due critic, gradient clipping e replay permanente delle demo completano la configurazione. Non riutilizzare checkpoint/dataset legacy: azioni 4, osservazioni 112 e schema `peg-sequence-sac-1` sono incompatibili con i precedenti 14/101.

| Parametro | Default training | Curriculum conservativo |
|---|---:|---:|
| Learning rate | 0,0001 | 0,00005 |
| Peso BC | 5 | 10 |
| Temperatura entropica fissa | 0,001 | 0,0003 |
| Gradient steps / vector step | 1 | 1 |
| Frazione demo | 0,5 | 0,5 |
| Batch / replay | 256 / 300.000 | 256 / 300.000 |
| Gamma / tau | 0,999 / 0,005 | 0,999 / 0,005 |
| Rete attore/critic | 256, 256 / twin critic | uguale |

Sono configurazioni motivate dai fallimenti precedenti, **non iperparametri già ottimizzati con nuovi risultati reali**. `tune.py` implementa un confronto controllato fra conservativo, bilanciato e più esplorativo: pilot da 25.000 transizioni ciascuno, selezione su 100 episodi seed 123 e verifica indipendente seed 999. Promozione solo con gate ≥80%; la scelta vale per la skill/guida testata.

Il curriculum visita le sette skill nell'ordine della tabella. Per ciascuna riduce la guida **1 → 0,5 → 0,25 → 0**, valuta prima di addestrare e procede solo con almeno 80/100 successi della skill. Se necessario esegue un budget di 100.000 transizioni, rivaluta e si ferma al primo gate fallito. Alla fine verifica `full`, guida zero, su seed 999 oltre al gate seed 123. I seed distinti controllano ripetibilità nello scenario fisso; non costituiscono domain randomization o prova di generalizzazione. I budget SB3 contano transizioni aggregate su tutte le istanze, non secondi né passi per singola istanza.

## Fisica mantenuta e limiti

La versione riusa geometria e condizioni di capture di `peg_transfer/sac_*`; la tenuta usa il servo isolato descritto sopra, senza alterare i vecchi checkpoint:

- Unità metri/radianti; passo fisico 0,005 s, quattro substep per azione. Origini delle istanze separate di 2,5 m. PSM e peg vengono ripristinati dalle pose iniziali configurate; `settle` attende 25 controlli prima dell'approach.
- Peg con foro passante composto da 36 settori convex, non un unico convex hull che chiude il foro. Foro raggio 3,5 mm, post raggio 2 mm, margine 0,2 mm; il controllo conservativo considera il poligono interno e circa 1,283 mm di tolleranza radiale. Board e post restano collider attivi. La precedente prova utente `logs/sac/scene_check.json` verificava L5/L6/R2 e rifiutava una posa sul muro del foro.
- Root peg a riposo z≈24,613 mm; bordo superiore post z=35,5 mm; quota di completa estrazione root≈51,113 mm. Lift verso z=60 mm; handover root `(0.020, -0.0075, 0.065)` m. R2 è il post `(0.042, 0.015)` m; root di posa compensata per l'offset del foro, x≈36,214 mm, y=15 mm, z≈24,613 mm.
- Aggancio PSM1: pinza realmente sotto 0,18 rad, tool entro 4 mm dal punto di grasp, orientazione peg quasi verticale e otto substep validi (40 ms). PSM2 richiede anche lift stabile, fonte liberata e otto substep di capture. Handover richiede apertura PSM1 sopra 0,38 rad e possesso PSM2 mantenuto per cinque controlli (100 ms).
- Finché trattenuto, il peg segue una posa aggiornata dal codice, con offset misurato, orientazione identità e velocità annullata. Non sono simulati forza di serraggio, attrito delle dita come causa della presa, coppie trasmesse, compliance, slittamento o contatti bilaterali tra pinze e peg. Il ricevente viene riconosciuto tramite capture/latch, non una seconda presa fisica a forze.
- L'aggiornamento assistito è limitato a 4 mm/substep e controlla il segmento percorso rispetto a board/post, compreso il ripristino traslazionale dopo un disturbo. Prima dell'estrazione conserva XY del peg sul post; allungamento oltre 8 mm rompe l'aggancio. Dopo piccoli disturbi viene ripristinata l'orientazione identità; la proiezione angolare non è una simulazione di un vincolo a forze. Queste protezioni riguardano il peg assistito: non sono una prova generale di assenza di collisioni tool/post o di contatti corretti dell'intero robot.
- Dopo il rilascio il peg è dinamico. La posa richiede foro allineato a R2, quota entro 0,7 mm, upright, velocità lineare sotto 10 mm/s e angolare sotto 0,1 rad/s per 25 controlli (0,5 s); il successo completo richiede anche entrambi i tool oltre 25 mm dal peg e tutta la sequenza.
- CCD su GPU viene ignorato da Isaac Sim, come nei log utente; il controllo del segmento assistito resta necessario. Raggiungibilità PSM2, inserimento e posa sono stati verificati nella configurazione assistita fissa L5/R2 (baseline 3/3 e rete BC 100/100); restano da verificare nuove pose, perturbazioni e presa a forze.

Manifest, demo e checkpoint conservano fingerprint del codice e hash degli USD locali di peg/board. Lo USD PSM usa ancora l'URL remoto configurato in `simulation/utils/assets.py`; se non è presente localmente non viene aggiunto un falso hash dell'asset remoto. Una modifica del codice dopo la raccolta richiede di usare lo snapshot corrispondente o rigenerare demo/checkpoint; i vecchi risultati non vengono automaticamente marcati compatibili.

## Comandi brevi, nell'ordine

Nella workstation via SSH, con `(PegTransfer)` attivo, dalla root del repository. Copiare una riga per volta usando il pulsante copia del blocco; non serve esportare `PEG_CKPT`.

1. Test leggeri, senza Isaac:

```bash
bash peg_sequence.sh tests
```

2. **Prima prova decisiva:** tre trasferimenti completi con il controllore analitico. Non allena una rete. Salva report, episodi e traccia a 50 Hz:

```bash
bash peg_sequence.sh probe
```

Il report è stampato come `[SEQUENCE] Report: ...`, in `logs/sac_sequence/baseline_full_<timestamp>.json`; la traccia ha suffisso `.trace.jsonl`. Per proseguire servono `complete=true`, `episodes=3` e `full_success_rate=1`. Se fallisce, inviare report e traccia: correggere la tappa indicata prima di raccogliere demo o avviare training.

La stessa prova con video registrato sulla workstation, senza display SSH:

```bash
bash peg_sequence.sh probe --video
```

Il video viene scritto in `logs/sac_sequence/sequence_videos/<nome_report>/`. Non viene visualizzato automaticamente sul Mac; scaricarlo con lo stesso host SSH già usato, mediante `scp`, oppure aprirlo sul desktop della workstation. Per una prova grafica dalla workstation usare `probe.py --baseline ...` omettendo `--headless`; SSH senza display deve mantenere headless.

3. Solo dopo baseline 3/3: raccolta di sei trasferimenti completi reali, due per ciascuno dei tre livelli di guida:

```bash
bash peg_sequence.sh collect
```

Devono riuscire tutti e sei. Il comando stampa `[COLLECT] Verified demonstrations: ...full_demos_<timestamp>.npz`. Il dataset completo include entrambi i PSM e il place, non solo lift. Se una traiettoria fallisce il report viene salvato, l'archivio non viene prodotto.

4. Imitazione iniziale della nuova rete, senza SAC online:

```bash
bash peg_sequence.sh bc
```

5. Valutazione completa della rete con guida motoria zero:

```bash
bash peg_sequence.sh eval
```

`bc` sceglie l'ultima demo compatibile; `eval` sceglie l'ultimo `final.zip` compatibile in `logs/sac_sequence`. Entrambi stampano il percorso effettivo prima di partire. Non selezionano checkpoint SAC v3/v4/v5 né PPO. Per scegliere un run precedente, passare il percorso reale come primo argomento, per esempio `bash peg_sequence.sh eval logs/sac_sequence/NOME_RUN_REALE/final.zip`; questo esempio richiede il nome reale stampato, non la stringa `NOME_RUN_REALE`.

6. Se la rete non soddisfa il gate completo, apprendimento SAC con curriculum per entrambi i PSM. **Può essere lungo: l'esecuzione è dell'utente**, nessuno script lo avvia implicitamente dal coding agent:

```bash
bash peg_sequence.sh curriculum
```

Seleziona automaticamente baseline riuscita, demo e checkpoint compatibili; stampa i tre percorsi. Salva `logs/sac_sequence/curriculum_<timestamp>/curriculum.json`. Arresto automatico al primo gate fallito, con il report corrispondente e il checkpoint. L'aumento indiscriminato dei passi non è il passo successivo a un gate fallito: esaminare la tappa, la traiettoria e la pinza. Per diagnosticare quel preciso checkpoint con traccia, usare:

```bash
bash peg_sequence.sh eval --num_envs 1 --episodes 1 --trace
```

Un pilot facoltativo sulla skill bloccata confronta tre configurazioni. Esempio sulla fase `lift` con guida 0,5:

```bash
bash peg_sequence.sh tune --skill lift --guidance 0.5
```

Per riprendere il medesimo curriculum dopo analisi, usare lo stesso output directory e gli stessi file/configurazione. Esempio di forma del comando, da completare con il directory reale:

```bash
bash peg_sequence.sh curriculum --resume --output_dir logs/sac_sequence/DIRECTORY_CURRICULUM_REALE
```

Ogni run conserva `manifest.json`, `env.yaml`, sorgenti, Git diff/status, log TensorBoard, `pretrained.zip`, `final.zip` e checkpoint periodici SAC. I nuovi checkpoint hanno manifest JSON accanto al `.zip`. Nessun valore di loss BC viene interpretato come successo fisico.

7. Inference dopo valutazione. Video headless via SSH:

```bash
bash peg_sequence.sh video
```

Inference grafica sul desktop della workstation:

```bash
bash peg_sequence.sh play
```

Entrambe forzano `full` e guida zero. Per un'ulteriore valutazione indipendente:

```bash
bash peg_sequence.sh eval --seed 999
```

## Metriche da inviare e file introdotti

Inviare **prima** il report/traccia della baseline completa, indicando se il peg viene preso da PSM2 e rilasciato su R2; il video aiuta a controllare collisioni e presa assistita. Dopo raccolta/training inviare il report collector, il percorso `final.zip` stampato, `manifest.json`, la valutazione a guida zero e, in caso di fallimento, `.episodes.jsonl` e `.trace.jsonl`. I report includono grasp PSM1, lift stabile, grasp PSM2, handover, placement, full success, intervalli Wilson al 95%, timeout/failure, movimenti bloccati, tappa finale e cause del fallimento. Distinguono baseline, guida residua e rete senza guida motoria.

File aggiunti per questa revisione:

- `peg_sequence.sh`: comandi brevi e selezione dei file compatibili.
- `.../tasks/surgical/peg_transfer_sequence/{__init__,env_cfg,env,mdp,assistance,geometry}.py`: task, azioni, tappe, osservazioni, reward, guardie, servo assistito e punto di presa PSM2 sul bordo opposto.
- `.../reinforcement_learning/sac_sequence/{common,adapter,policy,model,train}.py`: provenienza, SB3 adapter, attore sul goal e SAC con obiettivo congiunto BC.
- `.../reinforcement_learning/sac_sequence/{rollout,probe,collect,evaluate,play}.py`: baseline, demo complete, valutazione e inference/video.
- `.../reinforcement_learning/sac_sequence/{curriculum,tune,paths,test_cpu}.py`: gate, confronto iperparametri, selezione degli artefatti e 22 regressioni CPU.
- `PEG_TRANSFER.md`: questa documentazione. Le precedenti modifiche non committate, demo, media e checkpoint sono conservati.

Controlli: 22 regressioni CPU nuove; 17 test leggeri legacy; parsing statico dei nuovi moduli; CLI `--help` senza avviare Isaac; sintassi Bash. Il gate finale richiede valutazioni reali di `full` senza guida; nessun risultato della baseline, del surrogato CPU o di una sola fase sostituisce quella prova.

## Storico della pipeline SAC v0 — risultati e comandi precedenti

Le sezioni seguenti conservano il lavoro e i risultati precedenti. I loro comandi non sono il percorso consigliato per la nuova versione sequence-v1.

# Peg transfer con due dVRK PSM — pipeline SAC v0

Repository: `/home/francesco/Documents/GIGI/i4h-workflows`. Task nuova: **`Isaac-Peg-Transfer-Dual-PSM-SAC-v0`**. Nel primo curriculum il peg parte sempre da **L5** e PSM2 lo posa sempre su **R2**: le sedi restano fisse. PSM1 prende il peg e lo solleva; PSM2 lo riceve dopo l'apertura di PSM1; infine PSM2 lo trasporta e lo posa.

**Stato del 2 ottobre 2026:** la prova geometrica e la prova guidata aperto → chiusura → lift sono passate nelle esecuzioni utente. Le cinque demo L5 della revisione v3 sono riuscite, ma sia l'attore inizializzato con imitazione sia il successivo pilot SAC di 50.000 transizioni danno **0/100 grasp e lift**, con 100 timeout. Anche il tentativo v4 con BC più forte ha dato 0/100 grasp e lift. Il trasferimento completo non è appreso. Durante il lavoro dell'agente sono state fatte solo analisi offline e verifiche leggere.

## Risultato guarded v4: pausa del training e confronto degli attori

Checkpoint: `logs/sac/peg_transfer_dual_psm/2026-10-02_12-12-38_peg_sac_L5_guarded_v4/final.zip`. Valutazione utente: `logs/sac/eval_L5_guarded_v4.json`, 100 episodi completi su L5, seed 123, zero per tutte le metriche, 100 timeout, zero failure e blocked_moves. Il tentativo con `bc_weight=10`, `demo_fraction=0.5` e `gradient_steps=1` non ha superato il gate lift.

Audit CPU in `logs/sac/offline_L5_guarded_v4_audit.json`: 25.008 transizioni online, distanza minima PSM1 6,88 mm, nessuna osservazione entro il raggio di presa di 4 mm, nessuna presa o lift nei 16 episodi di training terminati. I campioni sono a 50 Hz e non escludono passaggi brevi tra due campioni; le metriche del task e la valutazione restano comunque zero. Sui soli stati delle demo, l'errore assoluto medio del comando pinza durante `reach` cresce da 0,098 nell'attore `pretrained.zip` a 0,464 nell'attore `final.zip`; aumentano anche gli errori dei comandi di traslazione. Questo documenta una deriva sui dati dimostrativi, non ancora la traiettoria deterministica in simulazione.

Le due diagnosi utente sono ora complete: `policy_probe_L5_pretrained_v4.json` misura una distanza minima di 5,80 mm al passo 457 con pinza già chiusa (0,0702 rad); nessun campione entra nel raggio di presa. `policy_probe_L5_final_v4.json` misura invece 83,23 mm al passo 9, 780 campioni con pinza chiusa e nessuna presa. La traccia finale si allontana poi a circa 250 mm. L'errore esiste già nell'attore imitativo e SAC peggiora la traiettoria. Il prossimo intervento deve risolvere l'avvicinamento/chiusura fuori traiettoria e la deriva dell'attore. Le cinque demo ripetono L5: aumentare ancora i passi SAC o ripetere soltanto le stesse traiettorie non è supportato da questi risultati. La sorgente resta L5 e la destinazione R2.

### Recuperi v5 e gate di sola imitazione

**Esito reale v5:** la raccolta utente `grasp_recovery_L5_v5.json` ha recuperato con successo tutti e tre i prefissi (350, 395, 457 passi). `lift_recovery_L5_v5.npz` contiene 3.073 transizioni: le 2.575 originali più 498 azioni esperte di recupero. Il checkpoint BC-only `2026-10-02_12-30-27_peg_L5_recovery_bc_v5/final.zip` ha però dato ancora zero grasp/lift su 100 episodi, tutti timeout (`eval_L5_recovery_bc_v5.json`). La diagnosi `policy_probe_L5_recovery_v5.json` trova distanza minima PSM1 88,66 mm al reset: non migliora mai. La traccia mostra azioni di allontanamento già entro 50 passi e movimenti non nulli negli assi che l'esperto manteneva fermi, inclusi rotazioni PSM1 e PSM2. Non riprendere training SAC o handover da questa rete.

`diagnose_peg_policy.py --mask_lift_unused_actions` consente ora un confronto controllato: azzera le rotazioni PSM1 e i sei comandi di movimento PSM2 e mantiene la pinza PSM2 aperta. Lascia invariati i tre comandi di traslazione e la pinza di PSM1. La traccia salva sia `action` applicata sia `raw_policy_action`; il report dichiara `action_mask`. È una prova diagnostica con azioni modificate, non una valutazione della policy originale, e non dimostra da sola quale sia la causa della deriva. Uso esclusivo della fase lift. Verificati in CPU il mascheramento e il fingerprint invariato; la prova in Isaac Sim resta da eseguire dall'utente.

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/diagnose_peg_policy.py --checkpoint logs/sac/peg_transfer_dual_psm/2026-10-02_12-30-27_peg_L5_recovery_bc_v5/final.zip --phase lift --source L5 --max_steps 800 --headless --mask_lift_unused_actions --output logs/sac/policy_probe_L5_v5_masked.json
```

Il collector supporta ora `--recovery_checkpoint`, `--recovery_steps` e `--append_demonstrations`. Prima esegue azioni deterministiche della rete v4 imitativa per 350, 395 o 457 passi veri; poi il controller esperto recupera. La durata dell'episodio e lo stato del task continuano dal prefisso, senza reset dei tool/peg né allargamento delle soglie. Un prefisso terminato, con presa già avvenuta o peg non verticale viene rifiutato. Se il tool è nel corridoio esterno (errore x/z inferiore a 2 mm, y tra -15 e -4,5 mm rispetto al punto di presa), il recupero riapre la pinza sul posto, avvicina con pinza aperta e chiude soltanto nel raggio ammesso. Da altri stati usa il percorso esperto completo. Le condizioni di lift/tenuta e le collisioni restano quelle del task.

L'archivio nuovo contiene le cinque demo originali e soltanto le transizioni esperte dei recuperi; le azioni errate del prefisso non sono etichettate come esempi da imitare. Tutti i recuperi richiesti devono terminare con successo e chiusura verificata nella zona di capture, altrimenti il report viene scritto ma l'archivio non viene salvato. Il successo del controller di recupero non è successo della policy. Questa raccolta è implementata e verificata staticamente, ma attende la prova in Isaac Sim dell'utente.

1. Raccolta breve di tre recuperi su L5, con nomi nuovi:

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/diagnose_peg_grasp.py --source L5 --approach_mode capture --phase lift --repeats 3 --stage_steps 600 --headless --recovery_checkpoint logs/sac/peg_transfer_dual_psm/2026-10-02_12-12-38_peg_sac_L5_guarded_v4/pretrained.zip --recovery_steps 350 395 457 --append_demonstrations logs/sac/lift_capture_L5_v3.npz --output logs/sac/grasp_recovery_L5_v5.json --demonstrations logs/sac/lift_recovery_L5_v5.npz
```

2. Solo se la raccolta riporta `passed=True` e salva l'archivio, imitazione senza aggiornamenti SAC online:

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/sac/train.py --phase lift --source L5 --num_envs 16 --seed 42 --headless --initialize_from logs/sac/peg_transfer_dual_psm/2026-10-02_12-12-38_peg_sac_L5_guarded_v4/pretrained.zip --demonstrations logs/sac/lift_recovery_L5_v5.npz --bc_steps 3000 --imitation_only --run_name peg_L5_recovery_bc_v5
```

3. Valutare il nuovo `final.zip` stampato dal comando su 100 episodi con `evaluate.py --phase lift --source L5 --num_envs 16 --episodes 100 --seed 123 --headless --output logs/sac/eval_L5_recovery_bc_v5.json --checkpoint <percorso-reale>`. Non avviare SAC online finché l'attore imitativo non supera il gate lift; il problema separato della deriva SAC è ancora da correggere e validare. Il codice di training, il fingerprint, le vecchie demo e i checkpoint v4 restano compatibili.

Per gli script SAC lanciati direttamente via SSH, rendere importabile il namespace `simulation` nella stessa sessione:

```bash
export PYTHONPATH="/home/francesco/Documents/GIGI/i4h-workflows/workflows/robotic_surgery/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

L'import fallito di `simulation.utils.assets` interrompeva il training prima della creazione del run; il codice di uscita 0 osservato durante la chiusura di Kit non attestava un training riuscito.

## Risultato v3 L5 e prossimo controllo

Checkpoint con sola imitazione: `logs/sac/peg_transfer_dual_psm/2026-10-01_16-35-53_peg_sac_L5_lift_v3/final.zip`. Checkpoint dopo training online: `logs/sac/peg_transfer_dual_psm/2026-10-01_17-12-19_peg_sac_L5_lift_v3_refine/final.zip`. Le valutazioni complete sono `eval_L5_lift_v3.json` e `eval_L5_lift_v3_refine.json`: tutte le sei tappe zero, nessun fallimento, nessun movimento assistito bloccato. Anche i 48 episodi terminati durante il pilot online non registrano grasp.

L'audit CPU del replay di 50.000 transizioni trova **distanza minima PSM1 dal punto di presa 16,92 mm**, zero campioni entro 12 mm o nel raggio di aggancio di 4 mm. Questo riguarda le osservazioni registrate a 50 Hz, non ogni substep PhysX a 200 Hz. Il confronto sui soli stati delle demo mostra anche una deriva dei comandi di apertura: in `above_side` il comando medio PSM1 passa da 0,997 dopo imitazione a 0,558 dopo SAC, mentre l'esperto comanda +1. Il dettaglio è in `logs/sac/offline_L5_v3_policy_comparison.json`. Questi dati dimostrano mancato avvicinamento nel replay e deriva dell'attore, ma non identificano da soli tutte le cause della traiettoria chiusa.

Non prolungare il run `refine` e non avviare handover/full: la diagnosi è completata e mostra regressione. Il nuovo `scripts/environments/diagnose_peg_policy.py` osserva un episodio deterministico della rete: pose dei tool e del peg, distanze ai due punti di presa, angoli reali delle pinze, azioni, condizioni di aggancio, contatori e sei metriche. Non applica azioni esperte e non modifica fisica, reward, reset o checkpoint. È fuori dal bundle del fingerprint SAC; i checkpoint e le demo v3 restano validi. Campiona prima di ogni azione a 50 Hz; le condizioni simultanee campionate non sostituiscono il dwell verificato dal task a 200 Hz.

La traccia completata per `peg_sac_L5_lift_v3` mostra la chiusura iniziare al passo 354, quando PSM1 è a circa 7,6 mm dal grasp. La pinza raggiunge la soglia chiusa al passo 367, ma il tool non entra mai nel raggio di 4 mm: minima distanza 7,52 mm al passo 357. Poi si allontana fino a 15,31 mm al timeout; il peg resta sul post.

La traccia di `peg_sac_L5_lift_v3_refine` è peggiore: il tool non scende sotto 88,66 mm, poi si ritrae fino a circa 145 mm dal grasp e sale a z≈0,155 m. Le azioni si saturano in direzioni incompatibili con l'avvicinamento. Insieme agli zero successi, questo indica che il pilot SAC ha cancellato la traiettoria imitata; non continuare da quel checkpoint e non avviare handover/full.

Il prossimo esperimento riparte da `pretrained.zip`, prima del pilot SAC, con maggiore peso BC e metà dei campioni replay riservati alle demo. È un test controllato per limitare la deriva dell'attore, non una correzione già validata. Comando nell'ambiente `PegTransfer`:

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/sac/train.py \
  --phase lift --source L5 --num_envs 16 --seed 42 --headless \
  --initialize_from logs/sac/peg_transfer_dual_psm/2026-10-01_16-35-53_peg_sac_L5_lift_v3/pretrained.zip \
  --demonstrations logs/sac/lift_capture_L5_v3.npz --bc_steps 3000 \
  --bc_weight 10 --demo_fraction 0.5 --gradient_steps 1 \
  --steps 25000 --run_name peg_sac_L5_guarded_v4
```

Valutare 100 episodi su L5, poi generare video e un trace di un episodio per `final.zip`. Se il risultato resta a zero, servono correzioni esperte sulle traiettorie in cui la policy si discosta; aumentare ancora i passi SAC non è giustificato dai log correnti. Il video del solo attore (`.../2026-10-01_16-35-53_peg_sac_L5_lift_v3/videos/sac/rl-video-step-0.mp4`) mostra un tool apparentemente vicino al peg; la traccia misura però 7,52 mm di distanza minima, oltre la soglia fisica simulata di 4 mm.

## Ricostruzione del progetto

- Nessun `AGENTS.md` trovato nella catena delle directory o nel repository.
- La pipeline preesistente usa PPO/RSL-RL e le fasi `lift`, `handover`, `full`. Le modifiche non committate di `scene_cfg.py`, `env_cfg.py`, `mdp.py`, `agents/rsl_rl_cfg.py` e degli script RSL-RL sono state lette e conservate. Anche media, log, checkpoint e gli altri file locali preesistenti sono conservati. La registrazione SAC viene aggiunta in `__init__.py`; questa guida sostituisce la precedente dopo averla esaminata.
- La valutazione v6 **riferita nel contesto utente** è 100/100 lift stabili, 0/100 handover e successi completi. Non è stato trovato un JSON autonomo di quella valutazione e non è stata ripetuta. I TensorBoard v6 confermano reward di presa/lift, terminazione della fase lift e reward ricevente/handover nulle: la terminazione `success` in quel training significava solo lift.
- Checkpoint v6 presenti: `logs/rsl_rl/peg_transfer_dual_psm/2026-09-30_14-37-51_peg_lift_stable_v6/model_798.pt` e `2026-09-30_15-04-50_peg_lift_stable_v6/model_1798.pt`.
- Esistono anche run PPO handover v7, quindi lo stato reale è successivo alla sola v6. Nel run `2026-10-01_10-53-39_peg_handover_colliders_v7`, checkpoint `model_4797.pt`, le ultime reward ricevente/handover/successo sono zero. Nel run `2026-10-01_12-30-07_peg_handover_colliders_v7`, ultimo checkpoint trovato `model_2700.pt`, i log arrivano all'iterazione 2793: rare reward di presa del ricevente, handover/successo ancora zero. Sono statistiche di training, non tassi di valutazione indipendente.
- `media/V4.webm` dura 10,892 s. Fotogrammi a 1, 4 e 8 s: PSM1 sposta il peg lateralmente fuori dalla board; PSM2 non riceve il peg. La scrittura assistita della posa è nel codice, non dedotta come contatto fisico dal video.
- I collider e il vincolo verticale aggiunti nella versione PPO non risultano validati da una prova geometrica indipendente. I run v7 non dimostrano l'assenza di attraversamenti.
- Un checkpoint PPO `.pt` non è un checkpoint SAC. I nuovi script rifiutano `.pt`, task/schema incompatibili e file senza manifest SAC prima di importare il simulatore.

## Primo run SAC: risultato e prossimo controllo

Checkpoint: `logs/sac/peg_transfer_dual_psm/2026-10-01_14-23-37_peg_sac_lift/final.zip`. Valutazione reale in `logs/sac/eval_lift_full.json`: 100 episodi, seed 123, 52 L5 e 48 L6, tutte le sei tappe zero, 100 timeout, zero fallimenti e zero movimenti assistiti bloccati. Un valore zero di `blocked_moves` qui non prova che un lift sia sicuro: non c'è stato alcun hold.

`logs/sac/scene_check.json` riporta `passed=true`: posa iniziale L5/L6/R2 e contatto della caduta fuori centro verificati dall'utente, 36 collider presenti, nessun falso successo. Il caso fuori centro ha ribaltato il peg dopo il contatto; questa dinamica era prevista. Il probe non verifica ancora raggiungibilità e presa degli strumenti.

Analisi offline dei log: **1.248 episodi di training, zero grasp in tutti**. Replay con 500.000 transizioni: PSM1 entro 4 mm dal punto di presa solo **17 volte** (0,0034%), entro 12 mm 2.109 volte. Nessuna delle 17 osservazioni vicine rispettava anche il vincolo di orientamento del peg (`norm(q_xyz)<0,005`). Queste sono osservazioni a 50 Hz, non una traccia di tutti i substep a 200 Hz. La temperatura SAC finale è circa 0,000323; la reward media degli ultimi 100 episodi è circa -7,363, vicino al costo di un timeout lift. Nel video `media/SAC1.webm` gli strumenti si allontanano e il peg resta sul post.

Il curriculum attuale non ha scoperto la prima transizione. Il budget di progresso massimo raggiunto non penalizza l'allontanamento dopo il miglior avvicinamento; inoltre si esplorano rotazioni e PSM2 durante una fase che richiede solo PSM1. Non va prolungato ciecamente questo run né avviato handover. Prima si misura se il punto di presa può essere raggiunto e mantenuto con la pinza realmente chiusa senza inclinare il peg oltre la soglia.

È stato aggiunto `scripts/environments/diagnose_peg_grasp.py`: una prova guidata breve che comanda le vere azioni IK/pinza di PSM1 su L5 e L6, tenendo PSM2 fermo e aperto. Nella revisione 2 l'avvicinamento dall'esterno e dall'alto è seguito da chiusura fuori dal peg, avvicinamento lento alla zona di aggancio, lift verticale e tenuta. Non teletrasporta il peg o gli strumenti, non allarga le soglie, non attribuisce tappe mancanti e non è una policy appresa. Il primo stage non raggiunto ferma quel caso e produce misure per distinguere limite IK/collisione, mancata chiusura e aggancio non ammissibile. Il caso usa un timeout di almeno 20 s, calcolato automaticamente per coprire tutti i budget di tappa più 2 s; non usa il timeout lift di 8 s. `--stage_steps` aumenta il budget di ciascuna tappa dopo l’assestamento; `--episode_seconds` imposta esplicitamente il timeout generale.

Dopo la preparazione del terminale della guida, esegui:

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/diagnose_peg_grasp.py \
  --source both --output logs/sac/grasp_probe.json
```

Nel primo report del probe, entrambi i casi si fermano a `above_side` dopo 150 step (3 s), con `reset=false`: è il limite della tappa, non il timeout della simulazione. Errori finali al waypoint: circa 21,6 mm su L5 e 35,0 mm su L6. Per concedere 600 step (12 s) a ogni tappa e 120 s all'episodio, senza sovrascrivere il primo report:

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/diagnose_peg_grasp.py \
  --source both --stage_steps 600 --episode_seconds 120 \
  --output logs/sac/grasp_probe_extended.json
```

Il probe prosegue subito quando una tappa riesce; questi sono massimi. Con il solo `--stage_steps 600`, il timeout viene dimensionato automaticamente a circa 86,5 s nella revisione 2 (74,5 s con la sequenza originale `--approach_mode open`). Il report distingue `stage_budget_exhausted`, `time_out`, `failure`, `success` e chiusura della finestra. Un reset per `failure` non viene disabilitato aumentando il timeout. La prova riguarda soltanto reach/grasp/lift di PSM1, non il trasferimento completo.

Il report utente `grasp_probe_extended.json` della sequenza originale raggiunge i waypoint su entrambe le sorgenti, ma esaurisce 600 step in `close`. Le ganasce arrivano correttamente a ±0,07 rad. In `reach` ci sono solo due osservazioni per sorgente contemporaneamente entro 4 mm e allineate, mentre le pinze sono ancora aperte. Continuando ad avanzare l'orientamento del peg supera la soglia; in `close` non ci sono osservazioni vicine, chiuse e allineate simultaneamente. L'errore finale di orientamento rispetto all'identità è circa 23,3° su L5 e 16,0° su L6: questo valore comprende anche la rotazione attorno a Z, non misura soltanto l'inclinazione.

La revisione 2 del probe (`--approach_mode preclose`, default) chiude le ganasce al waypoint esterno, 15 mm dal punto di presa, prima di avvicinarsi. Limita il comando di avvicinamento a 0,3 mm per step e mira 3,5 mm all'esterno del punto nominale, dentro il raggio originale di 4 mm. Appena è nella zona ammessa con pinza chiusa, ferma l'avanzamento e tiene la posa misurata durante il dwell di aggancio; solo `holder=1` completa `close`. La protezione geometrica e tutte le soglie del task restano invariate. È una correzione della traiettoria diagnostica da validare in Isaac Sim, non prova di una presa riuscita o di apprendimento SAC. Per la prossima prova, conservando i report precedenti:

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/diagnose_peg_grasp.py \
  --source both --approach_mode preclose --stage_steps 600 --episode_seconds 120 \
  --output logs/sac/grasp_probe_preclose.json
```

Il report include `probe_revision`, `approach_mode` ed errore angolare rispetto all'identità. Con `--approach_mode open` si può riprodurre la sequenza originale. Prima di altro training verificare `grasp_psm1=true` e `lift_stable=true` in entrambi i casi, e ispezionare un video della prova.

La prova utente `grasp_probe_preclose.json` ha `passed=true`: **2/2 casi guidati**, uno L5 e uno L6, con presa assistita e lift stabile. L'aggancio avviene a circa 3,95 e 3,94 mm dal punto nominale; la radice raggiunge circa z=0,05994 m in entrambi i casi. Nel report non ci sono aggiornamenti assistiti bloccati. Il video `media/SAC_pregrasp.webm` mostra il lift e la chiusura anticipata delle pinze. Non prova contatti bilaterali, forza di presa o una policy appresa: l'assistenza permette un aggancio per prossimità con ganasce già chiuse.

Nella revisione 3 si aggiunge `--approach_mode capture`: avvicinamento con pinze aperte fino alla zona ammessa, arresto alla posa misurata, chiusura, attesa dell'aggancio e lift. Conserva il target esterno di 3,5 mm, il comando lento di 0,3 mm per step e tutte le soglie/collisioni del task. Non continua ad avanzare fino al punto nominale come la sequenza originale. Per questa modalità `passed=true` richiede anche evidenza di arrivo aperto senza holder, seguito da chiusura e holder PSM1 nella zona ammessa. Le nuove misure `closure_started_in_capture_zone` e `closure_completed_in_capture_zone` verificano l'ordine osservato, **non il contatto fisico**. Il report dichiara `physical_grasp_verified=false` anche se il probe passa. La modalità `preclose` rimane disponibile per confronto; il checkpoint e il task SAC restano invariati. Prossima prova da eseguire dall'utente:

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/diagnose_peg_grasp.py \
  --source both --approach_mode capture --stage_steps 600 --episode_seconds 120 \
  --output logs/sac/grasp_probe_capture.json
```

Controllare report, traccia e video prima di usare queste traiettorie come dimostrazioni SAC. La chiusura in prossimità può ancora perturbare il peg o fallire: questa nuova traiettoria non è stata eseguita dall'agente in Isaac Sim. Per una presa basata sui contatti occorreranno misure sulle due ganasce, calibrazione di frame/orientamento e verifica di attrito/forza; una diversa sequenza del probe non sostituisce queste verifiche.

Invia `grasp_probe.json`, `grasp_probe.trace.jsonl` e un video della prova. La traccia contiene distanza, target error, angoli reali delle ganasce, orientamento del peg, dwell candidato, holder, movimenti bloccati e tappe. Se fallisce lanciare lo script, includi lo stack trace. Non è necessario inviare nuovamente i log già nel repository.

La prova guidata riuscita consente ora di raccogliere dimostrazioni per inizializzare SAC e usare una reward che penalizza l'allontanamento. La revisione è implementata nella sezione seguente, ma non ancora validata con una policy in Isaac Sim. Se la nuova raccolta o la rete falliscono, conservare report/video e correggere la causa prima di prolungare il training; le soglie fisiche non vengono rese più permissive.

Le prime revisioni del solo probe erano fuori dall'hash SAC e lasciavano caricabile il checkpoint originale, fingerprint `0d06ef06f6055e602e16ba3bc4aa5793ac05800064804567af3f039544738c4e`. La successiva revisione del training descritta sotto cambia reward, timeout e osservazioni: il vecchio ZIP è conservato, ma il controllo di compatibilità lo rifiuta con il codice attuale. Non usarlo per `--resume` o `--initialize_from`; una sua futura rivalutazione richiede il codice originale salvato nel run. Nessun checkpoint/log è stato sovrascritto.

## Prossima esecuzione: SAC lift con dimostrazioni

Il report utente `logs/sac/grasp_probe_capture.json` contiene **2/2 casi guidati riusciti**, chiusura dopo arrivo aperto verificata su L5 e L6, circa 10,30 e 10,98 s. Il video `/home/francesco/Screencasts/SAC_pregrasp2.webm` è stato ispezionato. Il timeout lift originale era 8 s: non conteneva questa traiettoria riuscita. I timeout attuali sono lift 16 s, handover 30 s, full 40 s. Le condizioni fisiche di aggancio e le sei metriche non cambiano. Questa evidenza riguarda presa assistita e lift, non handover, posa, apprendimento della policy o contatti reali delle ganasce.

La nuova raccolta salva transizioni reali `obs, action, next_obs, reward, done` in un `.npz` senza pickle, più metadata JSON con fingerprint, hash dell'archivio, asset, clock, sorgenti, intervalli degli episodi ed evidenza della chiusura. Questa prima tappa usa **L5 soltanto**, fase lift e timeout normale di 16 s: osservazioni temporali, reward e terminali coincidono con il training. Non impostare `--episode_seconds` durante la raccolta. Lo stato terminale è quello prima del reset automatico; non viene sostituito dall'osservazione del nuovo episodio. Solo una raccolta completa con tutti i casi riusciti produce l'archivio. I precedenti JSON non hanno le osservazioni e le azioni complete, quindi non possono diventare replay senza una nuova esecuzione.

Dopo l'attivazione dell'ambiente `PegTransfer`, eseguire nell'ordine:

```bash
# 1. Cinque dimostrazioni su L5. Usare nomi nuovi: l'archivio v2 contiene già
#    demo di L5 e L6 e ha un fingerprint precedente.
python workflows/robotic_surgery/scripts/simulation/scripts/environments/diagnose_peg_grasp.py \
  --source L5 --approach_mode capture --phase lift --repeats 5 --stage_steps 600 \
  --output logs/sac/grasp_collect_L5_v3.json \
  --demonstrations logs/sac/lift_capture_L5_v3.npz

# 2. Inizializzare l'attore e salvare senza rollout/training SAC online.
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/sac/train.py \
  --phase lift --source L5 --num_envs 16 --seed 42 --headless \
  --demonstrations logs/sac/lift_capture_L5_v3.npz --bc_steps 3000 --imitation_only \
  --run_name peg_sac_L5_lift_v3

# 3. Usare il percorso effettivo stampato come [SAC] Checkpoint.
PEG_CKPT=/percorso/stampato/peg_sac_L5_lift_v3/final.zip
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/sac/evaluate.py \
  --checkpoint "$PEG_CKPT" --phase lift --source L5 --num_envs 16 --episodes 100 --seed 123 --headless \
  --output logs/sac/eval_L5_lift_v3.json
```

`--imitation_only` inizializza Isaac Sim per configurare l'ambiente, ma non esegue step online; fa solo aggiornamenti supervisionati dell'attore. Non addestra i critici e non aumenta `num_timesteps`. Il run salva `pretrained.zip`, `final.zip`, replay inizialmente vuoto, manifest e `imitation.json`. Inference e valutazione usano solo l'attore della rete, nessun waypoint/controller del probe. Valutare prima l'attore inizializzato: una perdita di imitazione bassa non garantisce una traiettoria riuscita in loop chiuso. Le ripetizioni partono dalle stesse pose: non costituiscono una randomizzazione del dominio o una verifica di robustezza.

Con dimostrazioni si usa SAC SB3 nativo per i critici, l'entropia e i target, con **25% di campioni demo permanenti** e 75% replay online per minibatch. I campioni demo sono bilanciati per sorgente e tappa, così la lunga fase di avvicinamento non nasconde la chiusura. In questa prima tappa tutte le demo e gli episodi di training/eval sono fissati a **L5**. L'attore riceve un aggiornamento supervisionato aggiuntivo per ogni aggiornamento SAC; le traslazioni sono pesate di più perché i comandi sono piccoli. Una penalità supervisionata sulla deviazione standard limita inizialmente l'esplorazione anche nelle rotazioni e in PSM2; le 14 azioni e la loro log-probabilità SAC restano presenti, senza maschere nascoste. `learning_starts=0` evita di ignorare l'attore inizializzato per 10.000 azioni uniformi casuali. Entropia iniziale `auto_0.001`. Non è SAC puro: è SAC con imitazione supervisionata e replay di dimostrazioni. I pesi e la quota demo sono salvati nel manifest.

Il pilot SAC online di 50.000 transizioni, ancora fissato a L5, è già stato eseguito e ha dato zero grasp. Il comando seguente documenta la procedura usata: non rilanciarlo prima della diagnosi indicata all'inizio. Il gate di almeno 80/100 lift stabili serve per passare a handover, non è un requisito per iniziare il training SAC online dalla rete imitativa.

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/sac/train.py \
  --phase lift --source L5 --num_envs 16 --seed 42 --headless --resume "$PEG_CKPT" \
  --steps 50000 --run_name peg_sac_capture_v2_refine
```

Ripetere la valutazione su L5 con il nuovo `final.zip`. La ripresa conserva attore/critici, optimizer, temperatura, replay e archivio demo originale; non rifà l'imitazione iniziale. Non modificare l'archivio, il codice, la sorgente o il numero di ambienti tra run e resume. Finché non arriva la valutazione della rete, nessun successo SAC è dimostrato. Dopo il gate lift, **non passare a `near` o `random`**: qui il curriculum deve aggiungere abilità mantenendo fisse L5 e R2. La fase successiva è il ricevimento da PSM2:

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/sac/train.py \
  --phase handover --source L5 --num_envs 16 --seed 42 --headless \
  --initialize_from "$PEG_CKPT" \
  --steps 1000000 --run_name peg_sac_L5_handover
```

Questa è la progressione di abilità con reset sempre da peg su L5 e destinazione R2: **(1)** `lift` insegna a PSM1 avvicinamento, presa e lift stabile; **(2)** `handover` insegna a PSM2 avvicinamento e presa, poi a PSM1 il rilascio, verificando che PSM2 tenga da solo il peg; **(3)** `full` aggiunge trasporto, allineamento, rilascio e posa stabile su R2, mantenendo l'intera sequenza. Ogni episodio riparte dallo stato iniziale reale: nessuna tappa viene simulata via reset. Il checkpoint precedente inizializza i pesi, ma non dimostra che la fase successiva sia stata appresa.

Dopo il gate handover, inizializzare `full` dal checkpoint handover e passare `--source L5` anche a training, valutazione e inference. Valutare i checkpoint in modalità `full` per verificare tutte le tappe. La randomizzazione delle sedi sorgente è un curriculum di robustezza separato: considerarla solo dopo una policy completa L5→R2 verificata; R2 resta fisso.

## Versioni e scelta SAC

Versioni lette dall'ambiente Conda `PegTransfer`: Isaac Sim **5.1.0.0**, Isaac Lab **0.48.0**, `isaaclab_rl` **0.4.4**, PyTorch **2.7.0+cu128**, Stable-Baselines3 **2.9.0**, Gymnasium **1.2.0**. Sono installati anche RSL-RL 3.0.1 e skrl 2.1.0.

Si usa **SAC nativo di Stable-Baselines3**, già installato, con actor gaussiano limitato da tanh, due critici, target Polyak ed entropia adattiva. Riferimenti: [SAC SB3](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html), [API VecEnv e osservazioni terminali](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html). Le firme e il comportamento sono stati verificati nei sorgenti locali della versione installata, inclusi `SAC.load`, replay, timeout e normalizzazione delle azioni.

Il wrapper SB3 locale di Isaac Lab etichetta come terminale l'osservazione già resettata e imposta ±100 quando lo spazio delle azioni è illimitato. La nuova `PegSACVecEnv` espone **Box(-1, 1, 14)** e riceve l'osservazione autentica e le metriche **prima** del reset asincrono. I timeout consentono bootstrap; successo/fallimento sono terminali senza bootstrap. Non viene usata VecNormalize: le grandezze sono scalate esplicitamente nelle osservazioni e il replay mantiene una rappresentazione fissa.

## Geometria e limiti fisici

Tutte le grandezze del task sono in **metri**, quaternioni **wxyz**, giunti delle pinze in **radianti**. Il reset aggiunge l'origine di ogni ambiente anche sull'asse z.

| Grandezza | Valore della task SAC |
| --- | --- |
| Board | centro (0,05; 0; 0,005), dimensioni (0,25; 0,10; 0,01), superficie z=0,010 |
| Post | raggio 0,002, altezza 0,025, centro z=0,023; fondo 0,0105, sommità 0,0355 |
| L5 / L6 / R2, asse xy | (0; 0) / (0; -0,030) / (0,042; 0,015) |
| Basi PSM1 / PSM2 | (-0,07; 0; 0,15) / (0,07; 0; 0,15), orientamento identità |
| USD peg | `Props/PegBlock/block.usd`, metri/unità=1, asse Z, scala uniforme 0,011 |
| Bound peg dopo tutte le trasformazioni USD | x=[-0,000926772; 0,015828740], y=[-0,009282851; 0,009283128], z=[-0,014613020; 0,002601549] |
| Centro foro relativo alla radice | (0,00578638; 0), misurato dalla sezione del mesh a z=-0,006 |
| Radice a riposo / reset | z=0,02461302 / 0,02501302, gap iniziale 0,4 mm sopra board |
| Radice per liberare il post | z≥0,05111302, margine verticale 1 mm |
| Radice su R2 | (0,03621362; 0,015; 0,02461302) |

Il mesh originale è chiuso e ha un foro lobato: raggio interno minimo circa 3,764 mm rispetto al centro misurato nella sezione. Il collider originale è `convexDecomposition`; la sola impostazione non garantisce che il foro venga preservato dal cooking PhysX.

La task SAC disabilita la collisione del mesh visivo e crea **36 prismi convessi** separati: esterno rettangolare con i bound originali, foro poligonale di raggio 3,5 mm e raggio inscritto minimo 3,483 mm. Nessun convex hull attraversa l'intero foro. Massa peg esplicita **5 g**, contact offset dei prismi **0,1 mm**, rest offset zero; board/post conservano contact offset **0,5 mm**. Il proxy esterno e il foro sono approssimazioni conservative del mesh visivo, non una replica esatta dei lobi. CCD viene abilitato sulla scena e sul peg. Solver peg: 16 iterazioni posizione, 4 velocità.

La presa rimane **assistita**, con queste semplificazioni precise:

1. Non richiede contatto simultaneo di entrambe le ganasce, attrito sufficiente, forza di presa o corretto orientamento dello strumento. Richiede apertura media misurata <0,18 rad e tool tip entro **4 mm** dal punto di presa per **8 substep PhysX (40 ms)**. I punti sono offset fissi dalla radice: PSM1=(3; -9; -6) mm, PSM2=(14; 0; -6) mm.
2. Durante il hold si scrivono direttamente posa e velocità del peg dopo ciascun substep a 200 Hz. Si mantiene orientamento identità e si annullano le velocità: gravità, inerzia, coppie e scivolamento non sono riprodotti durante la presa. Aggancio e aggiornamenti richiedono orientamento già entro circa 0,57° dall'identità; non vengono corretti ribaltamenti arbitrari.
3. Movimento massimo assistito: 4 mm per substep. Un controllo del percorso con 33 campioni e margine 0,2 mm verifica board e **tutti i 12 post**, inclusa la discesa su R2. Il test usa l'esterno rettangolare e il foro inscritto. Recupera solo piccoli errori di contatto della board entro 0,2 mm, finendo sopra la superficie. Una proposta non valida non viene applicata; il contatore `blocked_moves` la registra.
4. Prima di liberare il post sorgente, xy del peg resta sull'asse originario; allungare l'aggancio oltre 8 mm lo rompe. Un movimento laterale non può portare il peg attraverso la parete del cilindro tramite l'assistenza.
5. PSM2 può agganciare solo dopo lift stabile e liberazione del post. Durante il doppio grasp comanda ancora PSM1; PSM2 deve restare vicino e chiuso. All'apertura di PSM1 (>0,38 rad) l'offset di PSM2 viene acquisito dalla posa corrente, senza salto al tool ricevente. All'apertura di PSM2 il peg torna alla sola dinamica PhysX, con velocità iniziale azzerata dall'ultimo hold.
6. La collisione fisica strumenti/peg, board e post resta attiva, ma l'assistenza non è un vincolo meccanico con reazioni di contatto. Non è una misura della robustezza di presa di un dVRK reale. Le collisioni tra strumenti, le articolazioni e la raggiungibilità richiedono ispezione in Isaac Sim; le self-collision dei PSM restano disabilitate dalla configurazione asset preesistente. L'USD dVRK viene caricato dall'URL preesistente o dalla cache Isaac Sim.

La protezione geometrica riguarda gli aggiornamenti assistiti. Il comportamento del peg rilasciato dipende da PhysX, CCD, cooking dei prismi e solver: l'assenza di tunnelling fisico non è stata certificata offline.

## Azioni, osservazioni, reward e curriculum

**14 azioni:** [Δx, Δy, Δz, Δrx, Δry, Δrz, pinza] PSM1, poi lo stesso blocco PSM2. IK relativo DLS: traslazioni fino a **6 mm** e rotazioni fino a **0,015 rad** per step a 50 Hz. Il frame del comando è quello usato dall'azione IK Isaac Lab, relativo alla base robot. Ogni pinza usa una sola azione continua: -1→±0,07 rad (chiusa), +1→±0,50 rad (aperta), 0→±0,285 rad. Velocità massima delle pinze 0,8 rad/s, effort limit 1 dalla configurazione locale; sono parametri simulativi da verificare visivamente.

Le osservazioni includono posizioni/velocità dei giunti, distanze ai due punti di grasp, posa/quaternione/velocità del peg, target R2, holder, latch del ricevente, offset di aggancio, sorgente, storia ordinata delle tappe, contatori dwell, potenziali massimi raggiunti, potenziale corrente, tempo e azioni precedenti. Con otto giunti per PSM sono **101 componenti** nella revisione con dimostrazioni (100 nella prima versione); la dimensione effettiva viene salvata nel manifest. Architettura identica nelle tre fasi SAC, diversa da quella PPO.

La reward usa shaping `0,999 * Phi(next) - Phi(current)`, con lo stesso gamma di SAC e pesi di potenziale per tappa 2/4/4/2/4/4. Allontanarsi riduce la reward; le differenze di potenziale scontate si cancellano lungo una traiettoria. Il potenziale diventa zero a successo/fallimento; ai timeout resta presente perché SAC fa bootstrap sulla vera osservazione terminale. Bonus una sola volta per le sei tappe: **4/16/6/10/15/60**, costo **0,02 per step**, penalità sulle azioni e **10** per fallimento. Il bonus lift è ora sufficiente a compensare il costo della traiettoria guidata di 10–11 s. I gate rispettano l'ordine delle tappe; le condizioni di successo sono indipendenti dai bonus. Esistono bonus intermedi deliberati per rendere apprendibile il curriculum; una reward alta non prova il trasferimento. I massimi storici restano nelle osservazioni ma non assegnano più reward di progresso.

| Metrica indipendente dalla reward | Evidenza richiesta |
| --- | --- |
| `grasp_psm1` | aggancio assistito di PSM1 |
| `lift_stable` | post liberato, holder PSM1, z=55–80 mm, velocità finita della posa <15 mm/s per 15 step = 0,3 s |
| `grasp_psm2` | ricevente agganciato dopo lift stabile, vicino e chiuso |
| `handover` | PSM1 aperto; PSM2 mantiene da solo il peg sopra i post per 5 step = 0,1 s |
| `placement` | storia handover completa, nessun holder/latch, pinza PSM2 aperta, entrambi gli estremi dell'asse del foro contengono l'asse R2 con tolleranza 1,283 mm, altezza entro 0,7 mm dal riposo, velocità lineare e finita <10 mm/s e angolare <0,1 rad/s, per 25 step = 0,5 s |
| `full_success` | posa ancora valida e tool tip di entrambi i PSM oltre 25 mm dalla radice peg |

Fallimento: peg sotto il piano di riposo di oltre 3 mm, sopra z=0,15, oppure fuori dall'area x=±0,18/y=±0,09. Reset: scena e robot al default, velocità zero, tutti i latch/contatori/eventi/potenziali azzerati solo per gli ambienti resettati. Nella pipeline SAC la sorgente predefinita è L5; `--source near` e `--source random` sono opzioni esplicite di robustezza, fuori dal primo curriculum. Il successo della fase termina subito l'episodio: lift stabile per `lift` (16 s), handover per `handover` (30 s), successo completo per `full` (40 s). La valutazione usa **full come default**, anche per checkpoint delle prime due fasi; impostare esplicitamente `--phase lift` per la prima verifica della rete inizializzata dalle dimostrazioni.

SAC: LR 3e-4, gamma **0,999**, tau 0,005, due critici, reti [256,256], batch 256, replay uniforme **500.000 transizioni totali**, warm-up 10.000 transizioni, 4 aggiornamenti per step vettoriale, temperatura `auto_0.1`, target entropy -14. Il replay è NumPy in RAM e i minibatch vanno alla GPU; 16 ambienti non moltiplicano per 16 la capacità. Con 100 osservazioni il buffer occupa circa **430 MB** più overhead; salvataggio finale include un file replay di dimensioni simili. Le sincronizzazioni CPU/GPU limitano la scalabilità rispetto a un replay GPU.

Il curriculum è manuale e verificabile sulle **stesse pose L5→R2**: `lift` insegna presa e sollevamento a PSM1; `handover` aggiunge presa del ricevente e rilascio coordinato PSM1/PSM2; `full` aggiunge trasporto, posa e rilascio su R2. Ogni fase riparte dalla sorgente reale, senza reset che fingono lift/handover già avvenuti. `--initialize_from` carica solo actor/critici SAC, crea nuovi optimizer/temperatura e svuota il replay quando cambia fase; raccoglie subito con l'actor precedente (`learning_starts=0`). Se serve imitazione, passare nuovamente `--demonstrations`: il replay permanente e l'aggiornamento supervisionato ripartono dall'archivio validato. `--resume` conserva optimizer/temperatura e replay della **stessa fase e sorgente**, con lo stesso numero di ambienti. I file ZIP intermedi non hanno replay; usare `final.zip` per resume. Riprendere dopo interruzione non riproduce lo stato vivo del simulatore né esattamente i generatori casuali.

## File della nuova pipeline

Prefisso task: `workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.tasks/robotic/surgery/tasks/surgical/peg_transfer/`.

| File | Funzione |
| --- | --- |
| `sac_geometry.py`, `sac_assets.py` | geometria in metri, percorso protetto, collider composto |
| `sac_actions.py` | comando continuo delle due ganasce con una azione |
| `sac_mdp.py` | presa assistita, osservazioni, reward, tappe e reset |
| `sac_env.py`, `sac_env_cfg.py` | loop PhysX, snapshot prima del reset, configurazione SAC |
| `agents/sac_cfg.py`, `__init__.py` | iperparametri e nuova registrazione Gym |
| `scripts/reinforcement_learning/sac/{train,play,evaluate}.py` | training, inference grafica, metriche quantitative |
| `scripts/reinforcement_learning/sac/{common,vec_env}.py` | CLI, manifest, checkpoint, adattatore SB3 |
| `scripts/reinforcement_learning/sac/demonstrations.py` | archivio verificato, replay demo permanente, inizializzazione e aggiornamenti supervisionati dell'attore |
| `scripts/environments/diagnose_peg_grasp.py` | prova guidata aperto → chiusura → lift e raccolta delle transizioni reali |
| `scripts/environments/diagnose_peg_policy.py` | un episodio della policy con trace di pose, distanza di presa e pinze misurate, senza modifiche al task |
| `scripts/reinforcement_learning/sac/{audit_geometry,test_lightweight,inspect_scene}.py` | audit offline, test CPU, prova scena da eseguire in Isaac Sim |

I checkpoint nuovi vanno in `logs/sac/peg_transfer_dual_psm/<timestamp>_<run_name>/`. `final.zip` è accompagnato da `final.json` e `final_replay.pkl`. Ci sono manifest, configurazione ambiente, versioni, hash delle sorgenti e degli USD locali, copia delle sorgenti incluse quelle non committate, Git diff/status, TensorBoard e `episodes.jsonl`. Non modificare codice o asset tra training e valutazione: il controllo di provenienza rifiuta hash diversi. Conserva la directory completa. Il manifest dichiara sempre che serve una valutazione indipendente.

La revisione con curriculum e dimostrazioni aggiorna `sac_geometry.py`, `sac_mdp.py`, `sac_env.py`, `sac_env_cfg.py`, `common.py`, `train.py`, `evaluate.py`, `test_lightweight.py`, `diagnose_peg_grasp.py` e questa guida; aggiunge `demonstrations.py`. Il fingerprint del codice corrente è `8185393283bf784e324892c34514545dbe423a2343ad6da541aa1995edef90ae`. Il default SAC fissa la sorgente L5; i due checkpoint v3 sopra riportati hanno questo fingerprint. La diagnosi della policy aggiunta il 2 ottobre non lo cambia. Checkpoint anteriori con fingerprint diverso non si possono riprendere con il codice corrente. Modifiche PPO preesistenti, USD, collider e checkpoint/log originali sono conservati.

## Comandi da eseguire nell'ordine

Questi comandi sono per il tuo terminale Bash; **nessuna simulazione o training sul task è stata eseguita durante questa implementazione**.

### 1. Ambiente e verifiche offline

```bash
cd /home/francesco/Documents/GIGI/i4h-workflows
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate PegTransfer
export PYTHONPATH="$PWD/workflows/robotic_surgery/scripts:$PWD/workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.tasks:$PWD/workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.assets${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=Y
export ACCEPT_EULA=Y
PEG_SAC_SCRIPTS=workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/sac

python "$PEG_SAC_SCRIPTS/audit_geometry.py"
python "$PEG_SAC_SCRIPTS/test_lightweight.py"
```

Le verifiche eseguite qui: audit USD offline (bound/unità/sezione/foro), **17 test CPU passando** nella revisione con dimostrazioni, breve aggiornamento SAC/imitazione su ambiente fittizio, replay con osservazioni pre-reset e maschere timeout, provenienza e replay permanente delle demo, identità dell'attore salvato/ricaricato con SAC nativo e ripresa con imitazione, posizioni sorgente/jitter e label metriche, differenza scontata dei potenziali e penalità sull'allontanamento, rifiuto PPO, authoring API USD/PhysxSchema in stage in memoria, compilazione Python, help delle CLI e controllo whitespace Git. La prova su ambiente fittizio verifica il software, non il robot o il trasferimento. Gli script `--help` non importano Kit.

### 2. Prima prova Isaac Sim, senza policy

```bash
python "$PEG_SAC_SCRIPTS/inspect_scene.py" --num_envs 1 \
  --output logs/sac/scene_check.json
```

Lo script prova L5/L6/R2 rilasciati e una caduta fuori centro sul bordo del foro R2; nessun caso deve generare successo senza storia. Controlla JSON, collider nel viewport e contatti: il peg deve posarsi con il post dentro il foro e restare sulla board; la prova fuori centro deve mostrare contatto con la sommità, non attraversamento. Un ribaltamento successivo nella prova fuori centro è fisicamente possibile. Verifica anche pose iniziali dei robot, assenza di compenetrazione tra strumenti e board e raggiungibilità dei punti di presa: il probe a pinze aperte non certifica il movimento degli strumenti. Se la prova fallisce, conserva output e video e correggiamo prima del training.

### 3. Smoke test SAC breve, poi training lift (sequenza originale; attendere la diagnosi del primo run)

```bash
python "$PEG_SAC_SCRIPTS/train.py" --phase lift --source L5 --headless --num_envs 16 \
  --steps 1024 --learning_starts 256 --buffer_size 4096 \
  --run_name peg_sac_smoke --seed 42

python "$PEG_SAC_SCRIPTS/train.py" --phase lift --source L5 --headless --num_envs 16 \
  --steps 500000 --run_name peg_sac_lift --seed 42

peg_lift_runs=(logs/sac/peg_transfer_dual_psm/*_peg_sac_lift)
PEG_LIFT_CKPT="${peg_lift_runs[-1]}/final.zip"
python "$PEG_SAC_SCRIPTS/evaluate.py" --checkpoint "$PEG_LIFT_CKPT" \
  --phase full --source L5 --headless --episodes 100 --num_envs 16 --seed 123 \
  --output logs/sac/eval_lift_full.json
python "$PEG_SAC_SCRIPTS/play.py" --checkpoint "$PEG_LIFT_CKPT" --num_envs 1
```

Il primo budget di 500.000 transizioni non ha ottenuto grasp: non rilanciare questi training prima della diagnosi guidata. I budget seguenti restano i comandi della prima strategia, non una previsione di convergenza. **Gate prima di handover:** `complete=true`, almeno 100 episodi, `lift_stable_rate≥0,80` su L5 e video coerenti. Gli altri successi possono ancora essere zero. Per riprendere la stessa fase:

```bash
python "$PEG_SAC_SCRIPTS/train.py" --phase lift --source L5 --resume "$PEG_LIFT_CKPT" \
  --headless --num_envs 16 --steps 500000 --run_name peg_sac_lift --seed 42
```

Dopo resume, riesegui le due righe che selezionano il checkpoint e la valutazione. Le opzioni buffer/warm-up/gradient del resume sono quelle del checkpoint salvato.

### 4. Handover, soltanto dopo il gate lift

```bash
python "$PEG_SAC_SCRIPTS/train.py" --phase handover --source L5 --initialize_from "$PEG_LIFT_CKPT" \
  --headless --num_envs 16 --steps 1000000 --run_name peg_sac_handover --seed 42

peg_handover_runs=(logs/sac/peg_transfer_dual_psm/*_peg_sac_handover)
PEG_HANDOVER_CKPT="${peg_handover_runs[-1]}/final.zip"
python "$PEG_SAC_SCRIPTS/evaluate.py" --checkpoint "$PEG_HANDOVER_CKPT" \
  --phase full --source L5 --headless --episodes 100 --num_envs 16 --seed 123 \
  --output logs/sac/eval_handover_full.json
python "$PEG_SAC_SCRIPTS/play.py" --checkpoint "$PEG_HANDOVER_CKPT" --source L5 --num_envs 1
```

**Gate prima di full:** almeno 100 episodi completi, `handover_rate≥0,60`, mantenimento del lift e nessun passaggio ottenuto attraversando i post o con il ricevente lontano. Non usare la sola reward o il `success` della fase per decidere.

### 5. Trasferimento completo, valutazione e video

```bash
python "$PEG_SAC_SCRIPTS/train.py" --phase full --source L5 --initialize_from "$PEG_HANDOVER_CKPT" \
  --headless --num_envs 16 --steps 2000000 --run_name peg_sac_full --seed 42

peg_full_runs=(logs/sac/peg_transfer_dual_psm/*_peg_sac_full)
PEG_FULL_CKPT="${peg_full_runs[-1]}/final.zip"
python "$PEG_SAC_SCRIPTS/evaluate.py" --checkpoint "$PEG_FULL_CKPT" \
  --headless --source L5 --episodes 100 --num_envs 16 --seed 123 \
  --output logs/sac/eval_full.json
python "$PEG_SAC_SCRIPTS/evaluate.py" --checkpoint "$PEG_FULL_CKPT" \
  --headless --source L5 --episodes 100 --num_envs 16 --seed 456 \
  --output logs/sac/eval_full_L5.json
python "$PEG_SAC_SCRIPTS/play.py" --checkpoint "$PEG_FULL_CKPT" --source L5 --num_envs 1
python "$PEG_SAC_SCRIPTS/play.py" --checkpoint "$PEG_FULL_CKPT" --source L5 --num_envs 1 \
  --headless --video --video_length 1000
python -m tensorboard.main --logdir logs/sac/peg_transfer_dual_psm
```

La valutazione produce JSON con conteggi, tassi e intervalli Wilson 95% per tutte le sei tappe, timeout, fallimenti, movimenti bloccati, seed, fase addestrata/valutata e risultati per sorgente; un file `.episodes.jsonl` contiene ogni episodio. Le metriche provengono dallo stato pre-reset e non dai pesi della reward. Un'interruzione o la chiusura della finestra dà `complete=false`. `play.py` usa sempre la sequenza full deterministica. I video vanno nella directory del checkpoint, `videos/sac/`.

## Risultati da inviare per la prima iterazione

Invia `scene_check.json`, eventuali errori del probe, un video con foro/post e strumenti visibili; poi il percorso del checkpoint SAC lift, `final.json`, `manifest.json`, `env.yaml`, `episodes.jsonl`, `eval_lift_full.json` e il suo file episodi, più il video deterministico da L5 e L6. Aggiungi TensorBoard (reward, actor/critic loss, entropia, tassi delle tappe) e il comando effettivamente eseguito. Per un errore includi lo stack trace completo. Non serve trasferire inizialmente l'intero replay da centinaia di MB.

Il primo tasso SAC reale è zero per tutte le tappe. Un lift riuscito in una futura revisione, una reward crescente o un successo del curriculum non autorizzano a dichiarare riuscito il trasferimento completo.
