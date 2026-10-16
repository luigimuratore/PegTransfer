# Peg transfer — risultati e prossime modifiche

## 2 ottobre 2026 — primo checkpoint completo - task success  but grasping not optimal

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

## 5 ottobre 2026 — v2: geometria e presa fisica

Esaminate mesh del peg e geometria delle dita. Il collider v1 aveva un contorno rettangolare, diverso dal triangolo arrotondato della mesh. Preparata una prova separata con collider ricavati dalla mesh, foro aperto, contatti con attrito e coppia delle ganasce limitata.

Due candidati: presa laterale sui vertici ad altezze diverse; presa dall'alto sulla parete con un dito nel foro. La presa richiede contatti su entrambe le dita e un lift mantenuto per 2 secondi, senza aggancio artificiale.

**Verificato:** 11 test CPU v2, 22 test CPU della sequenza v1, integrità del checkpoint1 conservato. Lo scarto geometrico radiale massimo campionato è 0,077 mm. **Da verificare:** nuove prese in Isaac Sim. Non è stato addestrato un nuovo modello. Il peg resta un corpo rigido con contatti elastici; la deformazione della gomma resta da calibrare.

Comandi e figura in [PEG_GRASP_V2.md](PEG_GRASP_V2.md). Evidenza dei controlli in [logs/grasp_v2/geometry_validation_2026-10-05.json](logs/grasp_v2/geometry_validation_2026-10-05.json).

Prime prove fisiche v2: vertice e parete terminano nella fase `above` dopo 1000 passi (10 secondi simulati), ancora a circa 42 mm e 23 mm dal target. Nessun contatto con il peg. Il limite predefinito viene aumentato a 6000 passi per fase e reso configurabile con `--stage_steps`; la tenuta richiesta resta di 2 secondi. Superati 13 test CPU dopo la modifica. La presa non è ancora verificata.

Prove successive: anche con 6000 passi la presa sulla parete resta in `above`, con errore 0,28 mm dal waypoint ma circa 29 mm dal punto di presa e orientamento bloccato a 0,116 rad. Eliminato il giro di 180° nella posa della parete e introdotta una traiettoria assoluta continua (25 mm/s in spazio libero, 5 mm/s vicino al peg), con anticipo limitato e rilevamento del blocco dopo 5 secondi simulati. Il log distingue waypoint e punto di presa e registra posizioni/velocità dei giunti. GUI ridotta a 25 Hz senza modificare fisica/controllo. Superati 18 test CPU, compresa l'interfaccia dell'IK Isaac; nuova esecuzione fisica ancora da verificare. Nessun nuovo checkpoint o successo di presa dichiarato.

Nuove prove dell'utente (`wall_psm1_2026-10-05_11-44-24-492301`, `vertex_psm1_2026-10-05_11-45-21-479726`): approccio rapido, poi arresto in `above` con yaw del polso a −30°, errori angolari 0,121 e 0,951 rad. Nessun contatto. La cinematica ricavata dagli USD riproduce entrambe le pose registrate con errore sotto 0,001 mm e conferma l'incompatibilità delle pose richieste. Preparati ingressi obliqui: parete 20° dalla verticale, vertice 30° dall'orizzontale con direzione di chiusura adattata al polso. Verificati 72 punti per ciascuna delle quattro combinazioni pinza/presa, mantenendo i limiti originali e un margine di 2°/1 mm. Il comando rifiuta prima di Isaac i percorsi non raggiungibili. Superati 24 test CPU; nuova presa fisica ancora da verificare. Evidenza in [logs/grasp_v2/kinematics_validation_2026-10-05.json](logs/grasp_v2/kinematics_validation_2026-10-05.json).

Le tracce `wall_psm1_2026-10-05_12-07-42-423750` e `vertex_psm1_2026-10-05_12-09-43-594205` mostrano approccio riuscito, ma presa non mantenuta: parete con circa 0,14–0,17 N prima della perdita dei contatti, vertice con chiusura completa senza pressione stabile. Preparata una correzione: preload di 0,35 N per dito per 0,4 s, coppia massima 0,015 N·m, attrito sperimentale 0,9 verificato nei materiali PhysX, lift a 3 mm/s con pausa se la pressione cala e interruzione rapida per perdita/slip. Nessun aggancio o comando sulla posa del peg. **32 test CPU superati; la nuova tenuta fisica resta da verificare in Isaac.** Nessun nuovo checkpoint appreso; checkpoint1 conservato. Evidenza in [logs/grasp_v2/grip_validation_2026-10-05.json](logs/grasp_v2/grip_validation_2026-10-05.json).

Prove dell’utente delle 14:24–14:26: interruzione prima dei comandi di movimento per `Peg actual friction differs from requested friction`; zero episodi raccolti. Uniformato il materiale anche sul collider originale disabilitato, aggiunta l’assegnazione dei coefficienti nel buffer PhysX se necessaria e la verifica finale con diagnostica dei valori. Corretto il rilascio dell’ambiente in caso di errore all’avvio. **34 test CPU superati; nuova esecuzione Isaac da verificare.** Evidenza in [logs/grasp_v2/material_startup_validation_2026-10-05.json](logs/grasp_v2/material_startup_validation_2026-10-05.json).

**Prima presa fisica riuscita:** la prova dell'utente sulla parete delle 14:37 completa 1/1 episodi con PSM1: lift misurato 34,9 mm, tenuta di 2 s, forze mediane 0,344/0,304 N, poi abbassamento e rilascio. Nessun aggancio assistito. Salvati [risultato e hash](logs/grasp_v2/wall_success1_2026-10-05.json) e codice originale della prova. Non è ancora un trasferimento completo o un nuovo checkpoint appreso.

Sul vertice delle 14:35, raggiungimento riuscito ma chiusura con un primo contatto asimmetrico e circa 2,5 mm di spostamento del peg. Preparati ingresso più profondo, centratura delle dita ricavata dalla mesh e correzione limitata della posizione della pinza quando un dito tocca prima. Parete e materiali conservati. **41 test CPU superati; 2148 campioni della parete riproducono le stesse decisioni del controllo precedente. La nuova presa sul vertice deve ancora essere provata in Isaac.** Dettagli in [PEG_GRASP_V2.md](PEG_GRASP_V2.md).

Prova del vertice delle 14:59: **0/1**, arrivo riuscito e pressione su entrambe le dita, ma peg spinto di 1,002 mm durante la chiusura; interruzione prima del lift. Non viene usata come dimostrazione riuscita.

Preparati raccolta delle prese wall fisiche, training BC offline e valutazione della rete in Isaac. La rete apprende traslazione e ganasce; scelta della presa, orientamento e fasi restano programmati. Rete 128+128, 4000 aggiornamenti, batch 256, learning rate 0,0001; almeno tre episodi completi, di cui uno riservato alla verifica. **45 test CPU superati**, incluso training sintetico breve e salvataggio/rilettura. Nessuna nuova raccolta fisica o policy v2 valutata; l'ipotesi wall PSM1 / vertex PSM2 e il selettore restano da verificare. Comandi in [PEG_GRASP_V2.md](PEG_GRASP_V2.md).

### 5 ottobre 2026 — sei prese wall riuscite; primo modello arrestato prima della presa

La raccolta dell'utente delle 15:39 completa **6/6 prese wall fisiche** e salva 17.951 campioni. La rete `wall_bc1` ha errore BC basso (0,00170), ma nella traccia delle 15:51 tutti e tre gli episodi restano in `above`, a circa 6,7 mm dal waypoint, senza contatti. Non è un problema di attrito: la rete comanda una velocità quasi nulla prima di arrivare al peg. È disponibile solo la traccia dell'inference, non il report JSON conclusivo.

Preparata `wall_bc2`: feedback di posizione programmato, rete che impara guadagni positivi e comando ganasce. Le 41 pause di pressione del lift non insegnano più un arresto lontano dal target. Riuso delle sei dimostrazioni consentito solo dopo verifica del codice fisico salvato. **50 test CPU superati** e controllo breve di 100 aggiornamenti sui dati reali; nuova inference in Isaac ancora da eseguire. Parete originale, `wall_bc1` e checkpoint1 conservati. [Dettagli e comandi](PEG_GRASP_V2.md).

### 5 ottobre 2026 — bc2 arriva al peg; correzione della pressione e uso di CUDA

La traccia dell'utente delle 16:11 conferma approccio completo di `wall_bc2`, ma 25 s in `close` senza lift: una delle dita oscilla intorno a 0,30 N e il comando della rete quasi smette di chiudere. Il report conclusivo non è ancora presente. È inoltre arrivato il precedente report bc1, che conferma tre arresti prima della presa.

Preparata `wall_bc3`: feedback programmato sulla forza, rete che impara guadagni positivi di risposta. Training e inference spostati sulla RTX 6000; batch 1024, dati e campionamento sulla GPU. Camera frontale inclinata verso la board. **54 controlli del codice e quattro verifiche CUDA reali superati**; test di 100 aggiornamenti e inference sul campione di pressione debole. Nessuna nuova tenuta fisica dichiarata; prima del trasferimento completo serve verificare lift e hold del modello in Isaac. [Evidenza CUDA](logs/grasp_v2/wall_bc3_cuda_validation_2026-10-05.json).

Completato anche il training `wall_bc3` sulla RTX 6000: **4000 aggiornamenti**, batch 1024, learning rate 0,0001, seed 42. Modello salvato in `logs/grasp_v2/wall_bc3/policy.npz`; errore BC sull'episodio riservato 0,001238. Rilettura e inference CUDA del modello finale verificate; il comando mantiene chiusura sul campione con pressione debole. [Verifica del checkpoint](logs/grasp_v2/wall_bc3_checkpoint_cuda_validation_2026-10-05.json). La prova fisica della rete resta da eseguire sulla workstation.

### 5 ottobre 2026 — checkpoint2: presa wall PSM1 appresa e verificata

Il report delle 16:36 conferma **3/3 episodi completi di `wall_bc3`**: avvicinamento, presa fisica, lift 34,7–34,8 mm, tenuta di almeno 2 s, abbassamento e rilascio. Slittamento durante la tenuta ≤0,0251 mm. Rimangono programmati scelta della presa, fasi, orientamento e feedback di pressione; la rete BC regola guadagni positivi.

Salvati [manifest e misure](milestones/checkpoint2-wall-bc3.json), modello e sorgenti originali. Camera corretta sul lato Y positivo. Registrato sulla RTX 6000 un ulteriore episodio completo riuscito, con rendering RTX e codifica NVENC: [video frontale](logs/grasp_v2/videos/wall_bc3_front_verified_2026-10-05.mp4), 29,84 s. Il replay legge il codice originale del checkpoint; il lavoro sul task completo non lo sostituisce.

Avviata la sequenza fisica completa wall/wall. La prova separata PSM2 supera presa, lift e tenuta; il rilascio con apertura 0,15 rad resta a contatto su un dito (~0,021 N). Il controller completo apre maggiormente e deve verificare il distacco reale. Il percorso nominale L5 → zona di transfer → R2 supera 96 campioni IK entro i limiti originali; handover e deposito devono ancora essere verificati nei report fisici.

### 5 ottobre 2026 — camera frontale, video verificato e prime prove del full task

Il nuovo video frontale riproduce `wall_bc3` con **1/1 episodio completo riuscito**. Modello e sorgenti originali verificati tramite hash; rendering RTX e codifica NVENC, 1280×720 a 25 fps. Questo episodio è aggiuntivo rispetto ai tre del report dell'utente. Il checkpoint2 resta conservato e riproducibile con i comandi nel README.

Eseguite **sei prove fisiche sulla RTX 6000, 0/6 trasferimenti completi**:

| Prova | Cambiamento | Punto di arresto |
|---|---|---|
| 1 | Presa PSM1 originale, wall PSM2 | Scivolamento PSM1 mentre PSM2 chiude |
| 2 | Chiusura PSM2 più lenta e bilanciata | Scivolamento PSM1 mentre PSM2 chiude |
| 3 | Presa PSM1 profonda 2 mm, PSM2 3 mm | Forza >1 N durante il lift PSM1 |
| 4 | PSM1 originale, PSM2 più profondo | Scivolamento PSM1 mentre PSM2 chiude |
| 5 | PSM1 segue piccoli cambi di posa del peg | Scivolamento PSM1; polso vicino al limite |
| 6 | Presa PSM1 profonda con chiusura bilanciata | Forza >1 N durante il lift PSM1 |

La presa PSM1 originale regge il lift e il trasporto, ma non la perturbazione della presa ricevente. Rendere la presa più profonda incontra resistenza già sul post di origine; la causa precisa deve essere isolata. Il controller mantiene i limiti di forza e slip e non comanda mai la posa del peg. **Nessun nuovo training full**, perché manca ancora una dimostrazione completa fisica riuscita. Prima si risolve l'handover, poi si raccolgono dimostrazioni e si addestra il controllo completo.

[Report e hash delle sei prove](logs/grasp_v2/front_video_full_task_audit_2026-10-05.json). Report e tracce individuali sono conservati; per le prove 5 e 6 è salvata anche la copia esatta dei sorgenti usati. Cinque controlli del nuovo controller superati, inclusi contatti opposti, limiti IK, deposito con foro allineato e report di un peg ribaltato. La riuscita fisica resta quella misurata nei report sopra.

### 5 ottobre 2026 — checkpoint3: trasferimento completo con sostegno temporaneo

Su richiesta dell'utente, provate semplificazioni sul peg prima del training completo:

| Prova | Risultato |
|---|---|
| Sostegno elastico limitato | Scivolamento PSM1 nel transfer |
| Due varianti di giunto fisso con forza di rottura finita | Scivolamento PSM1 nel transfer |
| Giunto ideale, controlli di tenuta ancora soltanto fisici | Arresto durante la chiusura PSM2 |
| Giunto ideale e mantenimento assistito esplicito | **Primo trasferimento completo: 1/1** |
| Vincolo soltanto durante handover, tolto prima della tenuta PSM2 | Scivolamento PSM2: 0/1 |
| Ripetizione della configurazione riuscita | **3/3 trasferimenti completi** |
| Episodio aggiuntivo con camera frontale e seed 123 | **1/1 riuscito, video registrato** |

PSM1 mantiene presa, lift di 35 mm e tenuta di 2 s senza sostegno. Il giunto sostiene poi il peg dalla fase di transfer fino all'inserimento su R2; passa a PSM2 soltanto dopo due contatti opposti e 0,4 s di pressione misurata. Viene eliminato prima del rilascio ricevente. Il successo verifica deposito libero per 1 s, foro correttamente sul post, altezza e velocità del peg, entrambe le pinze staccate. Nessun comando diretto sulla posa o velocità dell'oggetto.

Nei tre episodi: errore finale **0,298 / 0,412 / 0,433 mm**, forza massima per dito **0,623 N**, entro il limite 1 N. Le tracce verificano tutte le 20 fasi, i contatti riceventi e il deposito libero. Lo spostamento misurato al primo passo di attivazione del sostegno è al massimo **0,0053 mm**, malgrado l'avviso USD sui frame del giunto. Nel terzo episodio il sostegno consente il mantenimento con contatti non sufficienti: queste deroghe sono registrate e il risultato resta dichiarato assistito.

Salvati [report 3/3](logs/grasp_v2/full_assisted_validation3_2026-10-05.json), [audit delle tracce](logs/grasp_v2/full_assisted_validation3_audit_2026-10-05.json), [video completo](logs/grasp_v2/videos/full_assisted_front_2026-10-05.mp4) e [manifest checkpoint3](milestones/checkpoint3-full-assisted.json). Codice della prova, scena e modello PSM1 sono conservati separatamente per il replay; checkpoint1 e checkpoint2 rimangono disponibili.

### 5 ottobre 2026 — checkpoint4: controllo neurale della sequenza completa

**Risultato: 3/3 episodi completi L5 → R2 sulla RTX 6000**, errore finale del foro 0,40–0,42 mm. La forza massima è 0,65 N; il peg resta libero e stabile sulla board per 1 s dopo il rilascio.

Esportati 17.930 campioni dalle tre sequenze riuscite del checkpoint3. Training Behavior Cloning su CUDA: due episodi per apprendere, uno riservato al controllo degli errori; rete 256 + 256 neuroni, 6.000 aggiornamenti, batch 1.024, learning rate 0,0001 e campionamento bilanciato delle 20 fasi.

Il primo modello `full_bc1` aveva un errore di imitazione basso ma **falliva il lift PSM1 per scivolamento: 0/1**. Nel modello `full_bc2` la rete PSM1 `wall_bc3` viene conservata identica; la nuova rete controlla transfer, PSM2 e deposito. Sono guadagni appresi per movimento XYZ e ganasce: obiettivi, ordine delle fasi, orientamento/IK e controlli di contatto restano programmati.

Il vincolo fisso temporaneo resta attivo durante il trasferimento e viene tolto prima del rilascio. Le tracce verificano presa iniziale PSM1 senza sostegno e due contatti PSM2 con precarico stabile prima dell'handover. **Non è ancora un risultato senza vincolo, con scelta autonoma della presa o con posizioni randomizzate.**

[Report 3/3](logs/grasp_v2/full_bc2_validation3_2026-10-05.json), [audit](logs/grasp_v2/full_bc2_validation3_audit_2026-10-05.json), [controlli CUDA](logs/grasp_v2/full_bc2_gpu_checks.json), [manifest checkpoint4](milestones/checkpoint4-full-bc2-assisted.json). Pesi e sorgenti hanno una copia separata in `.checkpoint_runs/checkpoint4-full-bc2-assisted/model/`; la scena originale resta quella conservata nel checkpoint3.

Nel riferimento checkpoint3, la rete BC controllava soltanto la presa PSM1; handover e place erano programmati. Checkpoint4 apprende anche i guadagni per il resto della sequenza, mantenendo gli stessi obiettivi programmati e il giunto ideale. La presa PSM2 senza sostegno resta da migliorare. La configurazione è fissa L5 → R2, senza domain randomization; il peg resta rigido con contatti elastici.

Verificato anche il replay dalla copia conservata: **1/1 episodio completo aggiuntivo sulla RTX 6000**, codice e modello identici tramite hash al riferimento 3/3. [Report del replay](logs/grasp_v2/full_assisted_frozen_replay_2026-10-05.json), [audit](logs/grasp_v2/full_assisted_frozen_replay_audit_2026-10-05.json). Sei controlli del controller, verifica CUDA del sostegno, sintassi dei launcher e integrità dei salvataggi superati. Nessun training completo avviato in questa verifica.

### 6 ottobre 2026 — prima pipeline SAC senza controller di sequenza

Implementata una rete SAC con **14 azioni libere**: XYZ, rotazione e ganascia per entrambi i PSM. Nessuna fase, waypoint o presa comandata entra nell'actor. Restano IK e limiti generici di movimento; il vincolo sul peg è disattivato. Curriculum presa/lift, handover, deposito completo, con avanzamento solo dopo ≥90% di successi su almeno 16 episodi dello stesso checkpoint.

Convertiti 17.930 campioni del checkpoint3; due episodi per training e uno tenuto separato. I campioni assistiti servono all'imitazione, non al critic senza vincolo. Rete 256+256, BC 6000 aggiornamenti, SAC batch 1024, learning rate 0,0001, gamma 0,9995, entropia 0,00001, 1000 aggiornamenti iniziali del critic, replay CUDA.

**Verifica infrastruttura:** 16 ambienti sulla RTX 6000, 100 passi fisici senza errori; sette controlli CUDA superati, inclusi aggiornamenti reali di actor/critic, save/load, deposito fisico e gestione corretta dei timeout. [Smoke test](logs/grasp_v2/sac_free_smoke_v1/smoke.json), [controlli CUDA](logs/grasp_v2/sac_free_tests_2026-10-06.log).

Terminato l'esperimento `sac_free_lift_v1`: 100.000 transizioni, 16 ambienti, episodi fino a 40 s. **BC: 0/16 pickup, tutti per timeout. Dopo SAC: 0/16, con 10 arresti per errore fisico e 6 timeout.** Pesi, sorgenti e report sono conservati; verificato anche l'avvio dell'inference con quei sorgenti congelati. Non si avanza nel curriculum. [Prima verifica](logs/grasp_v2/sac_free_lift_v1/pretrained_evaluation.json), [verifica dopo SAC](logs/grasp_v2/sac_free_lift_v1/evaluation.json).

Diagnosi sulle osservazioni del terzo episodio: i piccoli errori BC nei comandi di velocità producono circa 6,7 mm di bias verticale cumulato; dopo SAC il comportamento si allontana ancora dall'insegnante. Questo calcolo è un controllo offline, non una nuova misura del rollout.

Terminato `sac_free_lift_v2`: comandi di posa relativi allo strumento e apertura assoluta della pinza, sempre scelti dalla rete; loss unica SAC+BC con peso BC 80. Inizializzazione BC estesa a 12.000 aggiornamenti; stesso budget SAC e stesse proprietà fisiche. Il limite generico di anticipo della posizione durante contatto passa da 0,5 a 1 mm. Sette controlli CUDA superati anche per questa variante. **BC iniziale: 0/16 pickup. Dopo 100.000 transizioni SAC: 0/16 pickup, 16 timeout di 40 s, nessuna presa e forza di contatto massima 0 N in tutti gli episodi.** Non si avanza all'handover. [Log](logs/grasp_v2/sac_free_lift_v2_startup.log), [prima verifica](logs/grasp_v2/sac_free_lift_v2/pretrained_evaluation.json), [verifica dopo SAC](logs/grasp_v2/sac_free_lift_v2/evaluation.json), [parametri](logs/grasp_v2/sac_free_lift_v2/manifest.json), [controlli CUDA](logs/grasp_v2/sac_free_tests_pose_2026-10-06.log). Checkpoint4 resta conservato.

Diagnosi aggiuntiva su CUDA, con i sorgenti e i pesi originali verificati: il modello dopo sola imitazione si avvicina fino a **20,9 mm** dal punto di presa wall PSM1 nei primi 10 s, poi si allontana; dopo SAC arriva soltanto a **35,8 mm** nei primi 20 s, chiude la pinza lontano dal peg e produce diversi comandi quasi saturi. Nessun contatto in entrambe le tracce. Sono prove parziali di movimento, non nuovi episodi completi di valutazione. L'imitazione iniziale falliva già; gli aggiornamenti SAC non hanno risolto il raggiungimento. Prima di un altro training occorre correggere e verificare il pickup in esecuzione. [Diagnosi BC](logs/grasp_v2/sac_free_lift_v2_bc_probe/policy_probe.json), [diagnosi dopo SAC](logs/grasp_v2/sac_free_lift_v2_probe_fixed/policy_probe.json). Verificati nuovamente gli hash: checkpoint4 e il candidato SAC finale sono invariati.


### 6 ottobre 2026 — curriculum BC/DAgger senza vincolo

Nuova pipeline separata dai checkpoint riusciti: la rete osserva stato fisico e destinazione e sceglie pose e pinze. Nessuna fase o traiettoria dell’insegnante è fornita durante l’inference. IK e limiti di velocità restano controlli generali del robot. Tutti i training e le prove fisiche usano la RTX 6000 Ada.

**Dataset: 48/48 pickup dell’insegnante**, con jitter ±0,5 mm e yaw ±2° su L5→R2; 96.932 transizioni. Prima raccolta DAgger: **13/16 pickup**, 3 timeout e nessun arresto per forza; 26.978 transizioni riuscite aggiunte. L’insegnante resta attivo durante queste raccolte: non sono valutazioni autonome. [Dataset iniziale](logs/grasp_v2/autonomy_pickup_data_v1/evaluation.json), [recuperi](logs/grasp_v2/autonomy_dagger_approach_v2/evaluation.json).

BC diretto e ricorrente non superano le prime prove autonome. Il candidato con scelta neurale del comportamento e teste di posa lineari ottiene **4/4 prese fisiche, 0/4 lift**, con 4 arresti per forza eccessiva. Encoder 256+256, 98 osservazioni fisiche, 14 comandi, batch 1024, learning rate 0,0001, 12.000 aggiornamenti. Le fasi dell’insegnante sono usate soltanto come etichette nel training. [Valutazione candidata](logs/grasp_v2/autonomy_linear_options_v2_eval/evaluation.json).

Trovato e corretto un errore nella conversione delle pose: il limite componente per componente cambiava la direzione dei movimenti lunghi. Le varianti corrette conservano gli stessi pesi in cartelle separate. È in prova anche un limite generale di forza sulla pinza, senza vincolo al peg. **15 controlli CUDA superati**; i controlli software non certificano il pickup. [Controlli](logs/grasp_v2/autonomy_cuda_tests_force_2026-10-06.log).

Prova dell’insegnante per il task completo senza vincolo: **0/2**, presa e lift PSM1 riusciti, nessuna presa PSM2. Non vengono usate dimostrazioni fallite per addestrare il task completo. [Prova handover](logs/grasp_v2/autonomy_full_teacher_v2/evaluation.json).

Il curriculum `autonomy_curriculum_v3` è avviato con i due dataset: training, prova su 32 episodi e conferma con un altro seed solo se supera il 90%. `status.json` separa il pickup verificato dal task completo. Checkpoint4 conserva pesi e codice originali. [Procedura](PEG_AUTONOMY.md), [stato esperimento](logs/grasp_v2/autonomy_curriculum_v3/status.json).


Ripresa dopo interruzione: nessun processo attivo; `autonomy_curriculum_v3` era fermo durante DAgger dopo 37/48 episodi, senza dataset completo. La valutazione del round0 è completa: **32/32 prese, 0/32 lift**, 16 errori fisici e 16 timeout. Lo stato dell’esperimento è aggiornato come interrotto, non come successo.

Il limite di forza con riapertura automatica non ha migliorato il candidato: **0/4 lift**, e ha fatto fallire il pickup dell’insegnante nella prova handover. È stato rimosso dalle nuove prove. Il controllo di velocità precedente è nuovamente verificato: **4/4 pickup dell’insegnante**. [Verifica del ripristino](logs/grasp_v2/autonomy_teacher_restore_v4/evaluation.json).

L’isteresi dei punteggi della rete da sola dà **0/4 lift**. Le tracce mostrano ritorni dalla posa di sollevamento alla chiusura. Nuovo candidato v6: stessi parametri neurali, più memoria delle transizioni ricavata esclusivamente dagli episodi di training. Questa memoria vincola i cambi tra comportamenti appresi; non viene presentata come politica priva di vincoli. **16 controlli CUDA superati**, inclusa esclusione degli episodi di validazione dal conteggio delle transizioni. [Controlli](logs/grasp_v2/autonomy_cuda_tests_transition_2026-10-06.log). Nuova prova fisica in corso, separata dai precedenti checkpoint.


Prova v6 della sola memoria delle transizioni: i quattro PSM1 hanno perso i contatti durante il lift. Diagnosi interrotta dopo 2500 passi, senza dichiarare episodi completi riusciti. La variante v7 conserva la posa neurale all’ingresso in ogni comportamento, mantenendo il controllo neurale continuo delle pinze: **tre pickup iniziali realmente riusciti**, forza di picco 0,36–0,38 N.

Trovato un errore di campionamento nella valutazione parallela: i reset degli ambienti veloci potevano sostituire nell’insieme contato gli episodi lenti ancora aperti. Il report v7 è marcato con un audit esterno e non vale come promozione. V8 mantiene gli stessi pesi e parametri decisionali, ma completa ciascun episodio assegnato esattamente una volta. `autonomy_curriculum_v4` è avviato su **16 ambienti CUDA**: 32 episodi, conferma su altri 32 solo oltre il 90%, altrimenti 16 recuperi DAgger e nuovo BC.

Le nuove raccolte salvano progressivamente soltanto le dimostrazioni riuscite e complete, per resistere alle interruzioni. **20 controlli CUDA superati**, inclusi memoria delle pose, reset individuali, salvataggio dei dati e conteggio degli episodi veloci/lenti. [Controlli](logs/grasp_v2/autonomy_cuda_tests_sampling_2026-10-06.log), [audit v7](logs/grasp_v2/autonomy_linear_pose_memory_v7_eval/sampling_audit.json), [stato v4](logs/grasp_v2/autonomy_curriculum_v4/status.json).

Per l’handover è in prova anche la combinazione PSM1 wall→PSM2 vertex. È un esperimento dell’insegnante fisico senza aggancio al peg, non una capacità già appresa dal modello.

Ripresa verificata: il processo del curriculum v4 è ancora attivo; il tentativo vertex è terminato. **Wall PSM1→vertex PSM2: 1/1 handover fisico senza vincolo**, con entrambe le prese misurate e PSM1 rilasciato. Forza di picco 0,687 N; durata 40,73 s. L’insegnante resta attivo. [Report](logs/grasp_v2/autonomy_handover_vertex_v5/evaluation.json), [hash del risultato](logs/grasp_v2/autonomy_handover_vertex_v5/verified_result.json). Avviata la verifica del task completo su tre ambienti, senza vincolo, in `autonomy_full_vertex_native_v6`.

Nuovo training separato della pinza, mantenendo identici i pesi di movimento e scelta del comportamento. La variante v9 imita il cambiamento del riferimento articolare; la prova fisica segnala forza eccessiva e non viene promossa. V10 usa classificazione delle decisioni di pressione dell’insegnante ricostruite offline e calibrazione su forze controfattuali, esplicitamente distinte dalle dimostrazioni fisiche. Rete della pinza 64+64, lr 0,0003, 6.000 aggiornamenti, batch 1024. **23 controlli CUDA superati**, incluso salvataggio/caricamento e conservazione dei pesi di movimento. [Controlli](logs/grasp_v2/autonomy_cuda_tests_pressure_bc_2026-10-06.log).

Terminata la valutazione rigorosa v8: **21/32 pickup autonomi**, 32/32 prese, 11 timeout e nessun arresto per errore fisico; forza massima 0,390 N. Nessun insegnante né vincolo al peg. Il 65,6% non supera il gate del 90%: il curriculum raccoglie recuperi e ripete BC. [Report](logs/grasp_v2/autonomy_curriculum_v4/eval_round0/evaluation.json). Il training della pinza v10 è completato e la verifica fisica su 16 episodi è avviata.

Esito v9: **1/16 pickup**, 15 arresti per forza. V10 mantiene contatti ma non avanza al lift: diagnosi interrotta dopo oltre 15 s simulati, senza certificare episodi completi. V11 modifica soltanto la nuova rete della pinza, con fascia di pressione 0,36–0,40 N nelle etichette di training e decisioni discrete per evitare deriva durante la tenuta: **16/16 pickup autonomi**, nessun timeout o errore fisico, picco 0,429 N. [Report v11](logs/grasp_v2/autonomy_pressure_bc_v11_eval16/evaluation.json). Avviata conferma rigorosa 32+32 episodi con seed diversi in `autonomy_curriculum_v5`; il task completo non è ancora certificato.

La prova full wall→vertex rivela scivolamento poco dopo il rilascio di PSM1. Il precedente 1/1 handover era un passaggio momentaneo: un [audit separato](logs/grasp_v2/autonomy_handover_vertex_v5/handover_hold_audit.json) conserva il report originale ma chiarisce che non prova tenuta stabile. La diagnosi full viene interrotta con i peg sulla board e PSM2 senza contatti. Nelle nuove prove il solo PSM2 deve sostenere il peg sopra 37,5 mm per due secondi, con contatti opposti stabili. Nuovo insegnante: pressione PSM2 0,45–0,55 N e rilascio graduale PSM1 a 0,02 rad/s, con recupero del sostegno quando PSM2 si indebolisce. Sono parametri dell’insegnante per raccolta dati, non un controller della rete. **25 controlli CUDA superati**. [Controlli](logs/grasp_v2/autonomy_cuda_tests_transfer_interlock_v2_2026-10-06.log). Verifica fisica full avviata in `autonomy_full_vertex_slow_release_v7`.

### 6 ottobre — checkpoint5: pickup autonomo verificato

V11 supera entrambe le conferme: **31/32 con seed 123, 31/32 con seed 9876**, 62/64 (96,9%). Un arresto per forza in ciascun gruppo, zero timeout. BC gerarchico con movimento conservato e nuova rete della pinza; nessun insegnante o aggancio al peg durante l'inference. Variazioni ±0,5 mm e ±2° su L5. Non è SAC e non certifica ancora handover o placement.

Pesi, sorgenti e report copiati e verificati in `.checkpoint_runs/checkpoint5-native-pickup-bc`; [manifest](milestones/checkpoint5-native-pickup-bc.json). Replay: `bash peg_grasp_v2.sh checkpoint5-play`; video frontale anche da SSH: `bash peg_grasp_v2.sh checkpoint5-video`. Checkpoint4 resta invariato.

Il rilascio lento v7 non risolve la tenuta di PSM2. V8 estende l'inseguimento misurato di PSM1 durante il rilascio: osservato ancora scivolamento. Si mantiene il criterio fisico di due secondi di sostegno del solo ricevente prima di dichiarare handover. Nessun training full viene promosso da un passaggio momentaneo.

Replay dalla copia conservata: **1/1 pickup**, senza insegnante né aggancio. [Video frontale](logs/grasp_v2/checkpoint5-video_20261006_155611_038655216/inference_front.mp4), copiato anche nel bundle checkpoint5. **26 controlli CUDA superati**. Nuova prova fisica wall PSM2 sulla parete inferiore destra (300°), raggiungibile e libera nella geometria campionata: `autonomy_full_wall_lower_v9`.

### 6 ottobre — estensione L5/L6 e presa wall PSM2

Le prove v8/v9 restano diagnosi interrotte: vertex scivola durante il rilascio, wall si blocca al primo contatto debole. Corretto l'insegnante: continua la chiusura lenta prima di centrare una presa realmente caricata. Nuova verifica full `autonomy_full_wall_loaded_center_v10`, senza vincolo al peg. La caduta sulla board prima dell'handover viene ora rilevata come fallimento. **28 controlli CUDA superati**, inclusi caduta, ritiro dopo scivolamento e gate per ogni post.

Avviata raccolta di 32 episodi fisici su L5/L6, ±0,8 mm e ±4°, e curriculum `autonomy_positions_curriculum_v1`: verifica del checkpoint5 nelle nuove condizioni, eventuale recupero DAgger e nuovo BC, conferma su due seed. Nessuna promozione anticipata; checkpoint5 rimane il pickup certificato e checkpoint4 il full assistito conservato.

La raccolta delle posizioni termina con **29/32 pickup dell'insegnante**, tre timeout, zero arresti per forza; L5 14/15, L6 15/17. Salvate **68.705 transizioni** di episodi riusciti. È un dataset di training, non una verifica della rete. Il curriculum avvia la prima valutazione autonoma nelle nuove condizioni.

La prova full v10 registra un fallimento completato e due casi bloccati: il peg si torce e il ricevente carica una sola ganascia. Prova v11: correzioni limitate di posizione e orientamento del ricevente durante chiusura/rilascio, arrivo più preciso e chiusura condizionata all'allineamento misurato. Il peg non riceve comandi di posa durante l'episodio. Nessun trasferimento certificato.

### 6 ottobre — supervisione delle pose e confronto del donatore fermo

V11 e V12 full terminano **0/3**, tre cadute ciascuna; V12 inserisce PSM2 verticalmente 4 mm sotto il bordo, senza ottenere una presa stabile. [Report v12](logs/grasp_v2/autonomy_full_wall_vertical_deep_v12/evaluation.json). Avviata V13 con gli stessi parametri e PSM1 fermo durante il contatto; nessun comando di posa al peg.

Il checkpoint5 sulle condizioni ampliate ottiene **22/32** (L5 15/16, L6 7/16), senza promozione. Recuperi dell'insegnante **14/16**, 34.203 transizioni riuscite. Il nuovo BC presenta blocchi nella scelta/posa del lift. Correzione: allenare la posa conservata dall'ingresso del comportamento, eliminando la discrepanza fra training durante il movimento e inference all'ingresso. Nuovo candidato separato `autonomy_positions_entry_pressure_bc_v2`: 12.000 aggiornamenti movimento, 6.000 pressione, verifica su 32 ambienti CUDA in `autonomy_positions_entry_verification_v2`. Secondo seed richiesto prima della promozione. **30 controlli CUDA superati**; 27 file del checkpoint5 verificati integri.


Diagnosi delle etichette: dieci lift interrotti seguiti da recupero nel dataset L5/L6 venivano usati come mete raggiunte. V3 li esclude dalla regressione delle pose, mantenendo i loro esempi per la scelta del recupero. Aggiunta calibrazione offline dell'apertura della pinza; non è una simulazione riuscita né un controller inserito in inference. Training e verifica automatica in `autonomy_positions_completed_pose_v3`, 32 ambienti CUDA e secondo seed solo dopo il gate per post. **32 controlli CUDA superati**. Esito fisico ancora da verificare.

Il curriculum iniziale ampliato termina senza promozione: checkpoint5 22/32 nelle nuove condizioni, candidato riaddestrato **5/32**, 27 timeout, zero arresti per forza (L5 5/15, L6 0/17). [Report](logs/grasp_v2/autonomy_positions_curriculum_v1/eval_round1/evaluation.json). Le correzioni V2/V3 sono verificate separatamente; il checkpoint5 originale resta integro.

### 6 ottobre — checkpoint6: full nativo 3/3, ancora con insegnante

V13 completa **3/3 sequenze senza vincolo al peg**, nessuna caduta o timeout. Risolta la presa PSM2 mantenendo PSM1 fermo durante il contatto, invece di inseguire gli spostamenti del peg. Presa wall PSM2 a 300°, verticale, profondità 4 mm, apertura 0,25 rad; pressione 0,36–0,42 N. Handover: solo PSM2 sostiene sopra 37,5 mm per 2 s. Inserimento fino a fondo peg 30 mm, rilascio e assestamento fisico libero a 10 mm. Durate 59,93 / 61,41 / 60,91 s simulati; errore finale foro 1,07 / 1,22 / 1,30 mm; forze finali delle quattro ganasce nulle. Picco massimo 0,683 N.

[Report](logs/grasp_v2/autonomy_full_wall_fixed_donor_v13/evaluation.json); [checkpoint6 conservato](milestones/checkpoint6-native-full-teacher.json), 27 file con hash, incluso dataset di 18.225 transizioni riuscite. È un controllore delle dimostrazioni, **non un risultato autonomo della rete**. Avviata raccolta di 16 full nativi, L5→R2 ±0,25 mm / ±1°, e primo BC full: 16.000 aggiornamenti, batch 1024, lr 0,0001, 19 comportamenti appresi, un episodio intero escluso dal training. Valutazione separata senza insegnante in `autonomy_full_native_bc_v1_eval3`.

Il candidato posizioni V2 termina 5/32, V3 13/32 (L5 7/18, L6 6/14), rispettivamente 27 e 19 timeout, zero arresti per forza. Nessuno viene promosso. Checkpoint5 resta il pickup verificato 62/64 nelle sue condizioni originali. La diagnosi breve L6 dura 10 s e non vale come prova completa: negli ultimi campioni la rete sceglie il lift; [audit](logs/grasp_v2/autonomy_positions_v3_L6_diagnostic/gate_audit.json).

### 6 ottobre — ripresa, pressione full e transizioni

Verificati i processi e tutti i 27 file di checkpoint5/checkpoint6. La raccolta full V14 salva per ora 10 successi su 11 episodi conclusi, 62.592 transizioni; cinque casi restano al placement. Il replay video singolo arriva al rilascio ma rimane fuori tolleranza: interrotto e conservato come diagnosi, senza risultato completo. V16 misura nuovamente il contatto peg–PSM2 dopo l'handover, senza comandi all'oggetto.

BC full V15: 24.000 aggiornamenti pose/scelta, lr 0,0001; 10.000 pinze, lr 0,0003; batch 1.024, RTX 6000, 13 episodi riusciti con validazione per episodio. Conserva i pesi del movimento nel fitting delle pinze, esclude giunti/velocità dalla scelta e calibra offline l'apertura PSM2. Diagnosi: resta al primo waypoint, quindi non promosso. V17 riusa l'anticipazione di 20 campioni del pickup riuscito per imparare a passare alla fase successiva. Sono etichette di training; nessun timer/insegnante aggiunto all'inference. 36 controlli CUDA superati.

La raccolta V14 termina **10/16**, una caduta e cinque timeout al placement. Salvate soltanto le 62.592 transizioni dei dieci episodi completi riusciti. La prova V16 resta separata e deve dimostrare la correzione del placement.

### 6 ottobre — recuperi continui e verifica full automatica

V16 termina **12/16**, contro 10/16 prima della misura aggiornata della presa; una caduta e tre timeout al placement. Nessun vincolo/comando al peg. Le candidate V17/V18 raggiungono tre prese valide ma si bloccano nella scelta del lift: interrotte come diagnosi, non presentate come valutazioni complete. La banda neurale 0,36–0,40 N non risolve da sola il gate.

L'alternanza DAgger a ogni step crea comandi contrastanti e registra un arresto per forza; conservata come diagnosi V19. V20 usa invece 1.200 step continui della rete e poi recupero esclusivo dell'insegnante, sei scene; tutte hanno già superato il lift. V21 attende soltanto episodi full completati con successo, congela copie dei dataset, allena 24.000 aggiornamenti BC e 10.000 pinze su GPU e verifica la rete senza insegnante: pilot 3/3, poi almeno 15/16 su due seed con identici pesi. Nessun nuovo checkpoint full neurale certificato finora. **38 controlli CUDA superati**.

### 6 ottobre — selezione causale dei comportamenti, V22

Verificati processi e hash dei checkpoint5/6. V20 conclude **4/6 full** con recupero dell'insegnante, due cadute all'handover. V21 conclude **0/3 senza insegnante**, fermo al raggiungimento della presa. Il dataset consolidato contiene 19 full fisici riusciti, con quattro episodi interi di validazione. V22 aggiunge un selettore neurale che usa misure fisiche, comportamento precedente e obiettivo scelto dalla rete; i 25 tensori precedenti di movimento/pinze rimangono identici. Le etichette dei cambiamenti sono causali, senza anticipare il contatto o il lift nelle misure. 10.000 aggiornamenti CUDA, lr 0,0001, batch 1.024; rete 256+128. **40 controlli CUDA superati**, prova fisica nativa di tre episodi avviata. Nessuna nuova promozione finché non passa la verifica.

V22 si blocca al primo waypoint: diagnosi interrotta. V23 elimina le coordinate assolute del robot dal selettore e calibra offline il riconoscimento della propria meta, mantenendo vere le misure fisiche. Ottiene **tre prese native e due lift stabili**, ma non completa il trasferimento; diagnosi interrotta e conservata. V24 elimina anche la direzione delle forze e l'altezza assoluta: usa intensità/opposizione dei contatti, pinze, progresso fisico ed errore rispetto alla propria posa. Nessuna modifica al peg o alle soglie di successo. **41 controlli CUDA superati**; training V24 avviato.

V24/V25 restano diagnosi negative: una prova supera 1 N e termina come fallimento, le altre restano nella chiusura. V25 elimina l'apertura assoluta delle pinze dal selettore, mantenendola nel controllo della pressione. V26 aggiunge calibrazione offline della pressione sulle sole transizioni di chiusura già osservate, conservando i flag fisici originali; non sono nuovi successi fisici. 10.000 aggiornamenti CUDA, lr 0,0001, batch 1.024, 19 episodi completi con quattro tenuti separati. Nei due stati registrati prima bloccati, V26 sceglie ora lift e reach PSM2. **42 controlli CUDA superati**, nuova prova nativa di tre episodi in corso; nessuna promozione dichiarata.


### 6 ottobre — ripresa V28, presa conservata

V26 e V27 terminano entrambe **0/3 full nativi**: tre prese PSM1, due lift stabili, due cadute e un timeout. V27 conserva tra i comportamenti gli obiettivi che nelle dimostrazioni restano fermi, ma non risolve l’handover. Nessuna candidata viene promossa.

V28 incorpora i pesi originali del **checkpoint5 (pickup 62/64)** per approccio, presa e lift. La rete full continua a scegliere i comportamenti e controlla il resto del task. È una composizione di reti BC: non è SAC e non aggiunge un insegnante o un aggancio al peg. I pesi conservati sono verificati identici; 27 file di ciascuno dei checkpoint5/6 verificati. **45 controlli CUDA superati**. La prova di tre episodi full è avviata; il riuso dei pesi non certifica automaticamente il risultato della nuova composizione.

```bash
tail -f logs/grasp_v2/autonomy_full_preserved_pickup_v28_eval3_startup.log
cat logs/grasp_v2/autonomy_resume_status.json
```


V28 resta una diagnosi incompleta: tre prese, un lift stabile e una caduta conclusa; due scene restano ferme senza contatto. Interrotta con audit del processo. **V29** conserva anche il selettore neurale del pickup: riusare soltanto i decoder non aveva conservato quando la rete cambiava comportamento. La continuazione full è ancora scelta dalla rete full, senza soglie di contatto o timer di task aggiunti. **46 controlli CUDA superati**, verifica fisica avviata.


**V29 conclude 0/3 full, ma 3/3 prese e lift stabili.** Tutti cadono durante la chiusura PSM2. La posa appresa cambia di circa 2,5 mm e 0,15 rad all’ingresso in chiusura: un movimento indesiderato sotto carico. V30 usa la posa precedente nelle opzioni con aggiustamenti piccoli nelle dimostrazioni di training (tolleranze 2 mm / 0,13 rad). Non introduce pose dell’insegnante durante l’inference; pesi neurali invariati e criterio di successo invariato. È una semplificazione della memoria di posa, da verificare fisicamente; non un nuovo successo dichiarato.


V30 conclude **0/3 full**, tre prese e lift stabili, tre cadute durante la chiusura PSM2. La sola posa ferma non basta. **V31** mantiene l’ancora di contatto raggiunta e segue soltanto il cambiamento delle pose predette dalla rete rispetto all’ingresso nell’opzione, limitato a 2 mm / 0,13 rad. Le opzioni interessate sono ricavate dagli aggiustamenti presenti negli episodi di training; pickup conservato, nessun comando al peg. Pesi neurali invariati. **47 verifiche CUDA superate**, prova nativa avviata.


V31 conclude **0/3 full**: tre prese/lift, una presa PSM2 e un handover con sostegno del solo ricevitore per 2 s; due cadute e un timeout nella scelta del ritiro. Non promosso.

**V32/V33:** la posa del ricevitore è appresa nel riferimento del peg dai 15 episodi di training riusciti, con quattro episodi separati di validazione. RMSE di arrivo su questi ultimi: 0,0104 mm, risultato offline, non successo fisico. La posa corretta sotto carico viene conservata nel passaggio di comportamento. V33 riallena soltanto il selettore full, includendo apertura e riferimento delle pinze per distinguere rilascio/ritiro: 10.000 aggiornamenti CUDA, lr 0,0001, batch 1.024; decoder, pinze e pickup congelati. **49 controlli CUDA superati**, prova full nativa di tre episodi avviata. Nessun nuovo checkpoint full certificato.

### Ripresa: handover nativo verificato nel pilot, deposito ancora aperto

V33 termina **0/3 full**, ma completa **3/3 prese PSM1, lift, prese PSM2 e handover mantenuti dal solo PSM2 per 2 s**. Tre timeout al placement; nessuna caduta. Report: `logs/grasp_v2/autonomy_full_relative_router_v33_eval3/evaluation.json`.

V34 apprende la posa del peg rispetto al target dai 15 episodi TRAIN, escludendo quattro episodi interi VALID; il decoder usa la presa caricata misurata per comandare soltanto il PSM. Nessun vincolo o servo sull'oggetto. **51 verifiche CUDA superate**. Pilot nativo di tre episodi avviato; nessuna promozione prima del risultato fisico.

**V34: pilot completato 3/3 full nativi**, senza insegnante, controller di task o vincolo all'oggetto. Centraggio finale 0,503 / 0,743 / 1,183 mm; fondo del peg a 10 mm; tutte le pinze libere. Durata simulata 47,96 / 48,52 / 51,30 s. Forza massima 0,551 / 0,964 / 0,782 N. Report: `logs/grasp_v2/autonomy_full_destination_decoder_v34_eval3/evaluation.json`. Avviate due conferme da 16 episodi, seed 123/9876, ±0,25 mm / ±1°, pesi identici; promozione ancora pendente.

La prima conferma V34 incontra due cadute durante l'approccio PSM2: non può superare il criterio 15/16. I risultati completi restano da acquisire. V35 prova feedback limitato a 2 mm / 0,13 rad sui cambiamenti delle proprie pose apprese relative al peg; tutti i pesi neurali e il decoder del deposito restano identici. Pilot separato avviato, nessuna promozione. **53 controlli CUDA superati**, compresa la verifica dell'archivio.

Il video V34 resta una diagnosi interrotta: handover fisico completato, ma peg libero fuori centro al deposito. Audit salvato prima del SIGINT del solo processo verificato. Non viene pubblicato come replay riuscito. V36 prova la raccolta dell'insegnante con rilascio a 18 mm invece di 30 mm, senza vincoli al peg o modifiche alle tolleranze; eventuali dimostrazioni serviranno a rifittare il solo decoder di deposito.

**V35 termina 3/3 full nativi nel pilot fisso**, nessun timeout o fallimento; errore finale 0,776 / 0,748 / 1,313 mm e picchi di forza 0,587 / 0,601 / 0,609 N. Il feedback relativo conserva tutti i pesi neurali; cambia soltanto due maschere della memoria di posa. Avviate le due conferme con piccoli spostamenti, ancora senza promozione.

V34 conclude la prima conferma **11/16 full nativi**: due cadute durante l'approccio PSM2 e tre timeout di centraggio al deposito; 16/16 prese/lift e 14/16 handover. Il secondo seed non viene avviato perché il criterio non è superato. Report completo conservato; nessuna promozione.


### Ripresa — conferma V35 e centraggio appreso V37

Verificati i processi interrotti e gli hash dei modelli; le simulazioni precedenti erano ancora attive sulla RTX 6000. V35 conclude la conferma seed 123 **13/16 full nativi**: 16 prese/lift, 14 prese PSM2/handover, due cadute durante la chiusura PSM2 e un timeout al deposito (errore 2,446 mm). Non promosso; secondo seed non avviato. L'errore di orientamento durante il reach è inferiore a 0,001 rad nei casi esaminati: non supporta l'ipotesi di una rotazione obsoleta.

V36, insegnante con rilascio basso a 18 mm, conclude **1/3 full**, tre handover ma due arresti per forza oltre 1 N durante l'inserimento. Variante scartata; nessun successo dell'insegnante è contato come autonomia della rete.

V37 allena soltanto una rete 64+64 per piccoli aggiustamenti PSM2 nel riferimento del peg, dalle forze reali e dalla geometria misurata. Correzione limitata a 1 mm; etichette dai comandi dello strumento negli episodi nativi riusciti, nessun comando al peg. Tutti gli 81 tensori precedenti sono identici, compreso il pickup conservato. BC CUDA: 6.000 aggiornamenti, lr 0,0001, batch 1.024, 15 episodi TRAIN e quattro VALID; RMSE VALID 0,0752 mm (solo risultato offline). **56 controlli CUDA superati**. Pilot nativo full di tre episodi avviato, diagnostica ogni 10 passi. Report e sorgenti in cartelle separate; nessuna promozione prima delle prove fisiche.


V37 termina **0/3 full**, tre prese/lift e tre cadute prima della presa PSM2. I log ogni 10 passi mostrano che la correzione abbassa il proprio target di 0,56 mm prima di contatti PSM2 e senza spostamento del peg: l'ingresso geometrico consente una risposta indesiderata al proprio movimento. Variante scartata.

V38 conserva il resto di V35 e allena la correzione usando soltanto forze nel riferimento del peg e posizioni delle pinze. Nessuna posizione dello strumento entra in questa piccola rete; il resto della policy conserva lo stato fisico. 6.000 aggiornamenti CUDA, 15 TRAIN / quattro VALID; RMSE VALID 0,1533 mm, non successo fisico. **57 controlli CUDA superati**, compresa l'indipendenza della correzione dagli spostamenti dello strumento o dell'oggetto. Tutti gli 81 tensori precedenti invariati; pilot nativo full avviato. Checkpoint5/6: tutti i 27 file di ciascun archivio ricontrollati e intatti.


### 7 ottobre — full nativo nel pilot e curriculum con presa assistita

Verificati file e processi prima di riprendere. V38 conclude **3/3 full nativi** nella prova fissa L5 → R2: 48–49 s simulati, errore finale 1,372 / 0,562 / 1,230 mm. Non è ancora una certificazione con posizioni variate.

Su richiesta dell’utente, V39 mantiene gli stessi pesi BC e aggiunge una molla virtuale alla presa: massimo 0,15 N / 0,002 Nm, attivata dai contatti misurati, senza imporre pose/velocità al peg. **3/3 full assistiti** nel pilot; vincolo rilasciato al deposito, picchi effettivi 0,121 / 0,110 / 0,107 N. Nessun insegnante o controller programmato della sequenza. **61 controlli CUDA superati.**

Avviato curriculum di verifica 100% → 50% → 25% → 10% → 0%, stessi pesi, 16 episodi per ciascuno di due seed a ogni livello, ±0,25 mm / ±1°. Passaggio soltanto con almeno 15/16 su entrambi; arresto al primo livello fallito. La rimozione dei vincoli viene verificata, non presunta. Vecchi checkpoint5/6 intatti. Report: `logs/grasp_v2/autonomy_full_assistance_curriculum_v39/status.json`.

Checkpoint7 archiviato separatamente: `milestones/checkpoint7-assisted-full-bc-pilot.json`. Conserva il pilot assistito 3/3, tutti i sorgenti e i parametri della molla. Replay tramite `checkpoint7-play` / `checkpoint7-video`, con verifica degli hash. Non sostituisce checkpoint5/6 e non certifica il full nativo.

V39, conferma seed123: errori osservati dopo il pilot fisso (una caduta con assistenza saturata a 0,15 N e un arresto per forza a 1,011 N durante l’inserimento). Atteso il report completo prima del conteggio finale. V40 conserva gli stessi pesi in una nuova copia dei sorgenti: presa assistita fino a 0,4 N / 0,004 Nm, guida temporanea di centraggio dopo il rilascio fino a 0,15 N / 0,002 Nm. Non comanda pose del peg e non sostituisce l’handover; il successo richiede che presa e guida siano spente. **64 controlli CUDA superati.** Pilot separato avviato; nuovo curriculum attende 3/3 prima delle conferme a due seed.

**V39 conferma completata: 13/16 full assistiti**, due fallimenti (caduta durante l’approccio PSM2 e forza oltre 1 N all’inserimento) e un timeout di deposito. Il curriculum si arresta al 100%; secondo seed e riduzione non vengono avviati. Pilot 3/3 e checkpoint7 restano validi nelle condizioni fisse dichiarate.

Replay video checkpoint7 concluso **1/1 full assistito**, centro del foro a 0,183 mm dalla destinazione e vincolo rilasciato. Camera frontale verificata visivamente; video 49,28 s / 1280×720 conservato in un archivio supplementare immutabile, senza modificare l’archivio del modello. Report: `milestones/checkpoint7-assisted-full-bc-pilot-video.json`.

V40 completa fisicamente **3/3 full assistiti**, ma la chiusura del simulatore fallisce per un buffer creato in `torch.inference_mode`. Risultato fisico conservato; nessuna promozione. Curriculum V40 fermato con audit del PID verificato. V41 conserva gli stessi pesi e gli stessi vincoli, mantiene i buffer persistenti e richiede anche una ricevuta di chiusura corretta. Regressione CUDA riprodotta prima della correzione, poi **64 controlli CUDA superati**. Nuovo pilot e curriculum avviati in cartelle separate.

**V41 pilot completato: 3/3 full assistiti, chiusura del simulatore corretta.** Durate 50,90 / 51,13 / 50,49 s; picchi delle pinze 0,520 / 0,613 / 0,621 N. La guida applica circa 0,012 N per 28 passi fisici (0,07 s) per episodio; holder=0 e guida disattivata al deposito. Archiviato come **checkpoint8-assisted-full-bc-constraints**, separatamente dai precedenti: pesi, 22 sorgenti e report/chiusura verificati. È un pilot fisso, non una certificazione nativa o con posizioni variate. Avviata automaticamente la conferma seed123 del livello100%; soglia15/16 per ciascuno di due seed prima di ridurre.

## 7 ottobre 2026 — confronto di velocità e randomizzazione sullo stesso piolo (V42)

Il curriculum V41 è terminato: **31/32 full assistiti al 100%** (16/16 e 15/16), **29/32 al 50%** (15/16 e 14/16). Riduzione arrestata; nessuna certificazione nativa. Il replay desktop dell’utente conclude inoltre **1/1 full assistito**, forza massima 0,542 N, senza insegnante. Checkpoint8, checkpoint7 e i rispettivi archivi sono verificati e immutati.

V42 avviato sulla RTX 6000, stessi pesi `64f0512c…`, nuovi sorgenti congelati in una cartella separata. Prima riferimento e confronto di velocità: 1,5× in aria/1,25× in contatto, oppure 1,5× solo in aria. Parametri di pinza, stabilità, limiti fisici e forza invariati. Promozione solo conservando ogni successo abbinato su due seed e riducendo la durata media almeno 2%. Questi sono candidati, non miglioramenti fisici già misurati.

Poi curriculum L5→R2 con pioli e board fissi: 0 mm/0°, ±0,25 mm/±1°, ±0,5 mm/±2°, ±0,75 mm/±3,5°, ±1 mm/±5°. La traslazione campiona ciascun asse XY; la rotazione avviene attorno al foro. Almeno 15/16 per ogni seed prima di ampliare. Vincoli al 100% costanti. Eventuali soli errori di approccio autorizzano un BC limitato alle pose PSM1 dei primi tre comportamenti: lr 2e−5, 2.000 aggiornamenti CUDA; tutti gli altri pesi identici e regressione sulle condizioni precedenti. Errori nel transfer/deposito fermano l’allargamento.

**70 controlli CUDA superati** (64+6), incluso un fine-tuning sintetico che verifica la protezione degli altri pesi; non costituisce un successo fisico. [Controlli](logs/grasp_v2/autonomy_cuda_motion_curriculum_checks_2026-10-07_v3.log). [Stato della simulazione](logs/grasp_v2/autonomy_motion_positions_v42/status.json).

V42: primo confronto fisico alla velocità originale completato **3/3 full assistiti**, con ricevuta di chiusura corretta. Durate simulate: 50.95 s, 51.30 s, 49.16 s. Ora conferma con perturbazioni su 16 episodi per seed; velocità maggiore ancora da valutare.

### V42 concluso — velocità originale mantenuta, curriculum arrestato al transfer

| Condizione | Seed 123 | Seed 9876 | Decisione |
| --- | --- | --- | --- |
| Velocità originale, ±0,25 mm / ±1° | 16/16 | 15/16 | Livello superato |
| 1,5× in aria / 1,25× in contatto, ±0,25 mm / ±1° | 13/16 | Non eseguito | Velocità respinta |
| 1,5× solo in aria, ±0,25 mm / ±1° | 11/16 | Non eseguito | Velocità respinta |
| Velocità originale, ±0,5 mm / ±2° | 16/16 | 14/16 | Livello respinto |

Le velocità maggiori passano entrambe il pilot fisso 3/3, ma perdono successi quando il peg è perturbato: non vengono adottate. Il livello ±0,5 mm / ±2° fallisce per un timeout e una caduta nella chiusura PSM2. Presa/lift PSM1 riusciti in tutti i 32 episodi; nessun fine-tuning dell'approccio eseguito. Soglia invariata: almeno 15/16 per ciascun seed. Assistenza firm al 100% e guida del deposito ancora attive; nessuna nuova certificazione nativa.

Verificati i nove report completi, i relativi hash e le ricevute di chiusura Isaac. Checkpoint7 e checkpoint8 verificati immutati. [Riepilogo verificato](logs/grasp_v2/autonomy_motion_positions_v42/completion_summary.json).

**V43 avviato:** replica diagnostica dei 16 episodi del seed 9876 a ±0,5 mm / ±2°, stessi pesi/sorgenti V42, stessa velocità e stessi vincoli. Registra osservazioni, obiettivi e azioni per ogni ambiente durante i cambi di comportamento e ogni 50 passi. Un'analisi CUDA separata ricostruisce le decisioni della pinza PSM2; non modifica pesi o inference. Nella prima parte riprodotta la caduta del caso15; nel caso1 osservata una decisione di mantenere l'apertura con forze PSM2 di circa 0,105/0,162 N e errore rispetto al proprio obiettivo di 0,0185 mm. Dato diagnostico parziale, non risultato finale della valutazione. Report completo e analisi attesi prima di decidere il training di recupero.

### V43 concluso, recuperi PSM2 V45 e candidato BC V46

**V43: 14/16 full assistiti**, 16/16 prese/lift PSM1, 15/16 prese PSM2, 14/16 handover; un timeout e una caduta. Ricevuta di chiusura corretta, hash del report `5746c18f…`. Il caso1 resta in chiusura: ultimo campione forze PSM2 0,11869/0,17529 N, comando neurale della pinza fermo, proprio obiettivo raggiunto a 0,035 mm; la presa non viene caricata abbastanza per il passaggio. Il caso15 perde il peg. [Analisi](logs/grasp_v2/autonomy_handover_diagnostic_v43/receiver_close_audit.json).

V44, prima raccolta di recuperi, interrotto con SIGINT durante il prefisso dopo aver identificato il disallineamento della fase privata dell'insegnante. Nessun dataset completo o risultato promosso. V45 corregge il passaggio al recupero: verifica lift/contatti reali PSM1, acquisisce le trasformazioni fisiche correnti e riparte da reach PSM2. Riferimenti privati delle pinze sincronizzati con i comandi eseguiti, evitando chiusura accumulata da comandi dell'insegnante mai eseguiti. Modifica esclusiva della raccolta; nessun flag di progresso sintetico e nessun insegnante all'inference.

V45 raccoglie16 episodi assistiti, L5→R2, ±0,5 mm / ±2°, seed9876, prefisso neurale2650 passi. V46 attende dataset completo con almeno8 successi e almeno2 episodi VALID separati, poi aggiunge una testa neurale per la sola pinza PSM2 nelle opzioni proprie10/11. BC2.000 aggiornamenti CUDA, lr5e−5, batch1024, etichette delle azioni reali successive al recupero; tutti i tensori preesistenti byte-identici. Il modello deve superare pilot3/3, regressione dei precedenti seed e due seed nuovi2468/54321 al livello.5mm/2°, almeno15/16 ciascuno. Vincoli firm/guida invariati, nessuna promozione nativa o sostituzione di checkpoint8.

**67 controlli CUDA superati**:64 regressioni +3 controlli di recupero/isolamento/salvataggio. Il fit sintetico da2 aggiornamenti verifica protezione dei pesi, non una performance fisica. [Controlli nuovi](logs/grasp_v2/autonomy_receiver_cuda_checks_2026-10-07.log). [Stato V46](logs/grasp_v2/autonomy_receiver_bc_v46/status.json).

**Correzione della raccolta — V47/V48:** V45 era configurato con la presa PSM2 predefinita (parete60°, profondità1 mm, tilt20°), diversa dalle tre dimostrazioni native usate dal checkpoint. Fermati V45 e il workflow V46 prima di qualsiasi training. V47 ripristina la ricetta effettiva: parete300°, profondità4 mm, tilt0°, apertura0,25 rad, donatore fisso durante il contatto, rilascio a30 mm e ricattura della trasformazione del ricevitore caricato. Pressione0,36–0,42 N e rilascio donatore0,02 rad/s invariati.

Il fitter verifica ora sia la ricetta delle dimostrazioni sia la posa locale di reach appresa dal modello; dati della parete opposta vengono rifiutati. Le regressioni CUDA aggiornate passano3/3, insieme alle64 precedenti. V47 raccoglie i recuperi e V48 attende dataset completo, poi applica lo stesso training isolato e le cinque verifiche fisiche previste. [Stato corrente V48](logs/grasp_v2/autonomy_receiver_bc_v48/status.json). Checkpoint8 resta immutato; nessun candidato V46 è stato allenato.

## 7 ottobre — V48 verificato, checkpoint9 conservato; confronto velocità V49

V48 completa tutte le cinque verifiche: fisso3/3; ±0,25 mm / ±1°:16/16 e15/16; ±0,5 mm / ±2°:15/16 sul seed2468 e16/16 sul seed54321. Pesi SHA256 `9a371d1f95ea2cefe7680949ab229426672e650e6cc58fe4fe6d2eaf64fa5d74`. BC2.000 aggiornamenti della sola nuova testa pinza PSM2; pesi precedenti protetti. Presa firm1 e guida al deposito attive; full nativo non verificato. [Stato V48](logs/grasp_v2/autonomy_receiver_bc_v48/status.json).

Conservato **checkpoint9**, pesi+codice congelato+report+ricevute di chiusura,47 file verificati. Replay `bash peg_grasp_v2.sh checkpoint9-play --episodes 3`. [Milestone](milestones/checkpoint9-assisted-full-bc-randomized.json).

Richiesta: velocizzare molto gli spostamenti, mantenendo lente le interazioni. Il precedente limite3 mm/s si applicava anche al peg già preso. **V49 avviato:** stessi pesi, nuovo profilo opzionale con libero50/75/100 mm/s e trasporto isolato stabile9/12 mm/s. Contatto3 mm/s, pinze/hold/guadagni/limiti articolari/criteri invariati. Regime scelto dalla geometria e dai contatti misurati; nessuna fase dell'insegnante. Rampa di accelerazione e rallentamento presso peg, board, altro braccio o presa instabile.

**76 verifiche CUDA passate**;6 nuove includono conservazione bit-identica del profilo originale, trasporto con sola presa opposta stabile, rallentamento durante handover/inserimento, reset selettivi, pinze e stato fisico invariati. La validazione fisica richiede3/3 fisso, ≥15/16 per ciascun seed2468/54321 e ogni precedente successo conservato, con ≥2% di riduzione del tempo medio simulato. Nessuna nuova velocità ancora promossa. [Stato V49](logs/grasp_v2/autonomy_transit_speed_v49/status.json).

## 7 ottobre — V49 scartato; V50 traslazioni con rotazioni originali

V49 concluso: riferimento3/3 full assistiti; profili liberi2x/3x/4x con trasporto3x/4x, tutti0/3. Nove timeout: presa PSM1 raggiunta in ogni caso, nessun lift stabile. Report e ricevute di chiusura verificati, nessun profilo promosso. [Riepilogo negativo](logs/grasp_v2/autonomy_transit_speed_v49/completion_summary.json).

Il comando autonomy-transit-play rifiutava correttamente una velocità non verificata, ma mostrava un traceback poco chiaro. Ora distingue prove in corso/prove scartate e indica checkpoint9-play, con uscita2 e senza lanciare candidati falliti.

Avviato V50: nuove prove isolate di traslazioni libere50/75/100 mm/s e trasporto6/9/12 mm/s. Mantiene rotazioni originali1/0,2 rad/s e velocità minima25 mm/s senza contatto vicino al peg; il trasporto accelera solo con presa stabile e fondo peg già a43–50 mm. Pesi identici aV48, pinze/hold/criteri/vincoli invariati, nessuna fase dell'insegnante. Otto regressioni CUDA del controllo di velocità passate, incluse compatibilità esatta alle scale1, rotazioni e approach invariati e reset selettivi. Stesso gate fisico3/3 e≥15/16 per entrambi seed2468/54321, conservando ogni precedente successo e≥2% di riduzione media. [Stato V50](logs/grasp_v2/autonomy_transit_speed_v50/status.json).

## 7 ottobre — V50 verificato; checkpoint10 trasporto2× conservato

Concluso confronto fisico a pesi BC V48 congelati: baseline3/3; profilo2× fisso3/3, seed2468 15/16, seed54321 16/16, **31/32 conservando ogni precedente successo**. Il tempo medio simulato sui31 casi riusciti passa51,97→47,64 s, miglioramento8,3%. Libero fino50 mm/s, trasporto stabile fino6 mm/s, contatto3 mm/s; rotazioni, ganasce, tempi di hold e vincoli invariati. Non è nuovo training né SAC.

Il3× completa15/16 sul seed54321 ma perde un caso precedentemente riuscito: scartato anche se supera la soglia aggregata. Il4× completa16/16 sul2468 ma14/16 sul54321: scartato. Dieci report fisici e ricevute di chiusura verificati;78 controlli CUDA passati. [Riepilogo](logs/grasp_v2/autonomy_transit_speed_v50/completion_summary.json).

Salvato **checkpoint10**,52 file con pesi, codice e configurazione2× congelati, report e riferimenti abbinati. Replay: `bash peg_grasp_v2.sh checkpoint10-play --episodes 3`. Video: `bash peg_grasp_v2.sh checkpoint10-video`. Checkpoint9 originale ancora verificato e immutato. [Milestone](milestones/checkpoint10-assisted-full-bc-transit.json). Molla di presa e guida al deposito rimangono attive; full nativo non certificato. Il miglioramento8,3% riguarda durata simulata, non garantisce pari riduzione del tempo reale sotto rendering/carico concorrente.

## 7 ottobre — Video frontale checkpoint10

Registrato un episodio completo, **1/1 full assistito riuscito**, seed2468, ±0,5 mm / ±2°, profilo travel2× verificato. Video48,4 s,1280×720,25 fps,H264/NVENC; fotogramma della vista frontale ispezionato. Report fisico, SHA dei pesi/video e chiusura Isaac verificati. [Video](logs/grasp_v2/videos/checkpoint10_front_20261007.mp4). [Metadati](logs/grasp_v2/videos/checkpoint10_front_20261007.json). Checkpoint10 conservato e immutato.

## 7 ottobre — V51: curriculum di rotazioni, offset e cilindri avviato

Da checkpoint10, pesi BC identici e velocità travel2×: baseline ±0,5 mm/±2°, rotazione progressiva fino±30°, poi offset fino±1,1 mm per asse. I reset sono schermati rispetto al foro della mesh e al cilindro di raggio2 mm; nessuna sovrapposizione iniziale accettata. Introdotti i nomi dei12 cilindri della scena conservata; camera frontale: picking L appare a destra, deposito R a sinistra.

I successivi livelli multi-cilindro verificano esplicitamente ogni coppia e ciascun seed, senza nascondere coppie assenti dietro una media globale. Soglia15/16 per seed/coppia; stop al primo livello insufficiente. Nessun nuovo training o risultato fisico ampliato dichiarato: questa prima prova misura la generalizzazione e prepara i report per eventuale BC/DAgger.83 test CUDA passati e52 file checkpoint10 verificati. [Stato V51](logs/grasp_v2/autonomy_domain_v51/status.json).

## 7 ottobre — V51 fermato alla baseline; V52 diagnostico avviato

V51 conclude15/16 sul seed2468 e15/16 sul54321:30/32 full assistiti. Il secondo seed perde un precedente successo, perciò il gate si ferma prima di provare ±5° e gli altri livelli. PSM1 presa/lift32/32; un peg cade durante chiusura PSM2, un altro episodio supera1 N durante rilascio del donatore. Report/chiusure/SHA verificati; checkpoint10 integro.

Avviato V52 con pesi e codice archivio checkpoint10 sul seed54321 e logging dei contatti/comandi. Serve a distinguere variabilità della simulazione da una regressione del runtime prima del BC correttivo. Nessun nuovo training e nessuna espansione validata. [Riepilogo V51](logs/grasp_v2/autonomy_domain_v51/completion_summary.json). [Stato V52](logs/grasp_v2/autonomy_baseline_audit_v52/status.json).

## 7 ottobre — V52:16/16 sull'archivio; V53 rotazione±5°

Checkpoint10 originale ripete16/16 full assistiti sul seed54321; il caso9 conclude con picco0,847 N, mentre in V51 falliva oltre1 N. Report/SHA/chiusura verificati,52 file archivio integri. Una sola ripetizione non stabilisce la causa della differenza; V51 rimane documentato30/32 e nessun livello maggiore certificato.

Avviato V53: inferenza dal codice archiviato, cambiando soltanto yaw2→5°, offset±0,5 mm, velocità travel2× e vincoli invariati.16 episodi per ciascuno dei due seed2468/54321, soglia≥15/16 per entrambi. Nessun training in questa prova; contatti/comandi registrati per un eventuale recupero BC. [Stato V53](logs/grasp_v2/autonomy_archived_yaw_v53/status.json).

## 7 ottobre — V53:31/32 a±5°; checkpoint11 conservato; V54 avviato

V53:15/16 seed2468 e16/16 seed54321 con offset±0,5 mm/yaw±5°. Tutti i task usano i pesi e il runtime checkpoint10 originali; nessun nuovo training. Caso15 seed2468 fallisce nel rilascio del donatore per1,003836 N. Report/chiusure verificati. Checkpoint11 conserva il nuovo range in40 file SHA-verificati e aggiunge replay/video dedicati; checkpoint10 integro.

V54 parte da una copia isolata dell'archivio: solo una riga del parser cambia per ammettere±30°/±1,1 mm;29 sorgenti e pesi invariati. Verifica ancora±5°, poi aumenta yaw e infine offset, senza cambiare pinze/controllo/vincoli. Gate≥15/16 per seed e mantenimento dei successi V53 nel confronto del runtime; stop al primo errore. [Stato V54](logs/grasp_v2/autonomy_isolated_domain_v54/status.json).

## 2026-10-07 — Video checkpoint11 e chiusura V54

Video frontale salvato come `logs/grasp_v2/videos/checkpoint11.mp4`: episodio neurale full assistito riuscito, 47,52 s, 1188 frame, 1280×720 a 25 fps, NVIDIA NVENC. Pesi e sorgenti archiviati invariati; report, chiusura ambiente e SHA256 verificati. Da ora ogni nuovo checkpoint avrà un video con il suo nome.

V54 termina alla ripetizione iniziale ±0,5 mm / ±5°: 15/16 su entrambi i seed, con un precedente successo perso sul seed54321 per forza oltre1 N durante il rilascio PSM1. Nessun ampliamento del dominio o nuovo training. La valutazione V53 31/32 rimane conservata insieme a checkpoint11.

## 2026-10-08 — V55: recuperi e BC mirato al passaggio della presa

V54 resta30/32 a±0,5mm/±5°. Audit CUDA del caso fallito: PSM2 chiude ancora a−0,02rad/s con forze0,350/0,764N; poi supera1N. Avviata raccolta32episodi su seed55001, rete fino al passo2100 poi insegnante con ricentraggio al rilascio e alleggerimento sopra0,65N. Il lift deve essere realmente misurato. BC previsto solo per il decoder delle pinze PSM2,2000update CUDA/lr2e−5/batch1024/KL0,1; tutti gli altri pesi protetti. Test CUDA di isolamento, salvataggio/ricarica, split intero, teacher handoff e alleggerimento superati; non certificano la riuscita fisica. Promozione soltanto dopo3/3 e confronti yaw2/yaw5 su due seed conservando i precedenti successi; video checkpoint12 obbligatorio. Checkpoint11 intatto.

## 2026-10-08 — V55 interrotto; recupero dati e ripartenza V56

V55 si ferma per `Receiver recovery requires a measured stable donor lift`:16 episodi conclusi su32 richiesti,12 full riusciti e salvati, training non iniziato. Raccolta incompleta mantenuta come tale. Corretto il passaggio all’insegnante: attesa del lift misurato, esclusione degli ambienti esauriti e maschera delle azioni realmente eseguite. Cinque regressioni CUDA superate.

V56 riusa le12 traiettorie riuscite verificate contro reset, report terminali, sorgenti e chiusura Isaac:10 TRAIN e2 VALID separati per episodio. Avviato BC mirato PSM2 su RTX6000,2000 aggiornamenti/lr2e−5/batch1024/KL0,1, poi pilot3/3 e regressioni yaw2/yaw5 su due seed. Il checkpoint11 e il suo video restano intatti; checkpoint12 sarà conservato soltanto dopo tutti i gate e un video full riuscito. [Stato V56](logs/grasp_v2/autonomy_handover_refine_v56/status.json).

### 9 ottobre — tre picking e tre placement

Obiettivo: tre episodi consecutivi **L5→R2, L6→R1, L4→R3**. V56 conclude0/3 e non viene promosso. La rete originale completa L5R2 e L4R3 con aiuti temporanei, ma L6 rimane nella chiusura: diagnosi interrotta con audit, non3/3. Nuova raccolta misurata12/12 lift riusciti sui tre cilindri. Due correzioni delle vecchie righe del selettore restano bloccate e vengono scartate. La candidata attuale aggiunge una piccola rete BC della scelta close/lift (8–32–32–2,2000 aggiornamenti,lr0,0003,batch1024 sulla RTX6000), conservando tutti i98 tensori precedenti e le uscite delle altre scelte. Vincoli espliciti: molla di presa anticipata soltanto dopo contatto misurato, coppia limitata di mantenimento verticale, scarico della pinza quando la forza è eccessiva, guida di deposito. Nessun risultato full nuovo certificato prima della verifica fisica. [Log pilot](logs/grasp_v2/three_posts_pickup_bc_v60/full_contact_pilot3.log).

**Pilot V60 concluso3/3**, senza insegnante: L5→R2(47,17s),L6→R1(43,51s),L4→R3(66,31s). Errori finali del foro0,0015/0,0008/0,2265mm; ambiente chiuso correttamente e hash del report verificato. In corso conferma12 episodi a±0,1mm/±0,5° e poi video sequenziale dei tre percorsi dal modello congelato. La verifica è assistita e limitata a queste tre coppie, non una certificazione generale di domain randomization.

**Conferma V60 conclusa12/12**, quattro episodi per ciascuna coppia, a±0,1mm/±0,5°,seed59002; stessi pesi, criteri fisici e aiuti. Nessun fallimento o timeout. In registrazione i tre episodi consecutivi centrati,seed59003, da pesi e sorgenti congelati.

**Checkpoint12 conservato:** pilot3/3, conferma12/12 e video sequenziale3/3. Nel video: L5→R2(47,63s),L6→R1(45,09s),L4→R3(63,78s); errori finali del foro0,0005/0,2000/0,0009mm. Video156,48s,1280×720,25fps,H264/NVENC, interamente decodificato e visuale frontale verificata sulle tre partenze e sui tre depositi.49 file archiviati con SHA256, stessi pesi/sorgenti/configurazione; checkpoint10/11 integri. È BC con aiuti temporanei espliciti, senza insegnante in inference; non è una certificazione full nativa o di domain randomization ampia. [Report](milestones/checkpoint12-three-posts-assisted.json).

### 9 ottobre — V61: movimenti veloci, interazioni invariate

Stessi pesi checkpoint12, stesse tre coppie e stessi vincoli. Primo confronto GPU travel4×: **3/3** riusciti, durata media **48,74 s contro52,33 s** (−6,9% sul seed59001). In prova limiti maggiori per spostamenti liberi/trasporto isolato e rotazioni libere; presa, scambio e inserimento conservano i limiti precedenti.11 controlli CUDA passati. Nessun nuovo checkpoint promosso finché non passano confronto fisico, conferma12 e video frontale. [Stato V61](logs/grasp_v2/three_posts_speed_v61/status.json).

Le velocità nominali massime provate (300mm/s liberi,96mm/s trasporto e3rad/s rotazione libera) bloccano due approcci PSM2: diagnosi incompleta conservata, profilo scartato. Con100mm/s liberi, i pilot a96 e48mm/s trasporto completano3/3, ma le conferme mostrano approcci PSM2 immobili; anche questi profili vengono scartati prima della promozione. Non sono valutazioni complete0/12: i successi parziali e le interruzioni restano nei rispettivi audit.

V62 prova una modifica limitata: raddoppia soltanto gli incrementi di apertura già scelti dalla rete quando le pinze sono vuote e distano più di6mm dal peg, fino al riferimento0,28rad. Serve a mantenere l'apertura al passo con l'approach più rapido. Chiusura, apertura vicino al peg, limiti fisici dei giunti e vincoli restano invariati.11 verifiche CUDA passate; pilot e conferma fisica ancora in corso. [Stato V62](logs/grasp_v2/three_posts_speed_v62/status.json).

**Riserva V61 verificata:**100mm/s liberi e24mm/s trasporto stabile, pilot3/3 e conferma12/12 sugli stessi reset checkpoint12. Media conferma46,06s contro53,05s(−13,17%), ogni precedente successo mantenuto; report e chiusura verificati. Nessuna modifica all'apertura delle pinze in questo profilo. V62 a48mm/s con apertura vuota più rapida conclude pilot3/3 in38,80s(−25,85%), ma perde un peg nella conferma: questo profilo non viene promosso. Gli esperimenti proseguono a velocità di trasporto inferiore.

### 12 ottobre — checkpoint13: conclusione e verifica dei video

Il workflow V62 aveva concluso il9 ottobre prima dell'interruzione: conservato **checkpoint13** con100mm/s liberi,24mm/s di trasporto stabile e apertura accelerata solo per pinze vuote lontane dal peg. Stessi pesi checkpoint12, nessun training aggiuntivo. Pilot3/3, conferma12/12 sugli stessi12 reset, video consecutivo3/3. Durata media conferma **46,30s contro53,05s**, −12,7%; tutti i precedenti successi mantenuti. La riserva V61 senza accelerazione dell'apertura misura46,06s nello stesso confronto: entrambi i profili sono circa13% più rapidi; la differenza0,24s non dimostra un vantaggio dell'apertura accelerata.

Completata la verifica rimasta in sospeso:55 file dell'archivio integri, chiusure e report fisici SHA-verificati, entrambi i video interamente decodificati. Video frontale `checkpoint13_three_episodes.mp4`:143,68s,3592 frame1280×720 a25fps; durate dei tre task40,42/46,76/56,53s. Primo episodio anche in `checkpoint13.mp4`,40,40s. Controllate le tre partenze, i tre depositi e fotogrammi di scambio. Metadati collegano pesi e filmati tramiteSHA256; originali conservati. Checkpoint10/11/12 integri. Nessun training o valutazione rimasto attivo. Replay e download aggiunti alREADME. Presa, scambio, inserimento e aiuti fisici rimangono invariati; verifica limitata alle tre coppie e ai reset provati. [Milestone](milestones/checkpoint13-three-posts-fast-assisted.json).

### 12 ottobre — primo pilot BC + SAC residuale

Creato `checkpoint1.txt`, report breve per presentare il risultato tecnico checkpoint13. Implementata una pipeline separata: BC congelata, correzioni SAC limitate a 0,15mm del riferimento sugli spostamenti senza contatto; pinze e rotazione restano BC. Replay e apprendimento su CUDA, memorie BC incluse nell'osservazione SAC, terminali catturati prima dei reset. Quattro verifiche CUDA passate e baseline fisica a residuo zero **3/3** sui tre percorsi (40,53/39,03/55,72s simulati). Gli aiuti fisici rimangono attivi. Avviato il workflow `logs/grasp_v2/bc_sac_residual_v1`: baseline12, warm replay fisico, 512 aggiornamenti critic, 12.000 transizioni SAC, conferma12 e video frontale. Nessun nuovo checkpoint promosso e nessun miglioramento SAC ancora dichiarato. [Strategia e comandi](docs/BC_SAC_RESIDUAL.md).

**Esito V1:** baseline12/12 superata; raccolta BC a residuo zero5/6, con timeout dell'ambiente2 L4→R3 prima della presa PSM2. Il gate ha impedito gli aggiornamenti critic/SAC: nessun candidato, conferma o video nuovo. La chiusura Kit ha restituito codice0 nonostante l'eccezione; il supervisore ha quindi tentato di valutare un modello inesistente. Corretto il controllo degli artefatti e degli esiti del subprocess; due regressioni passate. Audit salvato separatamente, sorgenti e report originali V1 conservati. Prossimo passo: diagnosticare lo stallo di PSM2 mantenendo il gate fisico.

**Stallo corretto e rilancio V4:** riprodotto lo stesso episodio fallito; PSM2 raggiunge il waypoint esterno con errore circa0,002mm, ma PSM1 perde pressione e la chiusura continua peggiora il contatto. Il limite generico della pinza e i filtri di collisione non risolvono;900N/m sempre attivi perturbano altri scambi (due prove complete5/6), quindi vengono scartati. La correzione accettata attiva900N/m e recupera gradualmente l'ultima apertura stabile soltanto dopo un calo sotto0,25N su PSM1, con lift già stabile e ricevente senza contatto. Ripristina300N/m al contatto PSM2; collisioni, forza0,4N, coppia0,004Nm e criteri di successo restano invariati. Nessuna fase viene imposta e nessun successo scritto artificialmente. È un aiuto temporaneo esplicito, non una verifica nativa.

Preflight sul seed fallito61001 **6/6** completi, recupero della pinza soltanto nell'ambiente2 prima bloccato; conferma base seed59002 con±0,1mm/±0,5° **12/12**. Dieci controlli CUDA passati, incluso replay/aggiornamento SAC con observation249 e reset delle memorie. Tutti55 file checkpoint13 integri. Rilanciato `logs/grasp_v2/bc_sac_residual_v4`: raccolta reale BC,512 aggiornamenti critic,12.000 transizioni SAC, confronto12 con stessa configurazione e video frontale del candidato. Al momento del rilancio nessun nuovo modello SAC certificato. Prove e audit negativi conservati separatamente.


BC+SAC V4 conclusa: 12.000 transizioni online su GPU. Baseline BC assistita 12/12; candidato BC+SAC 11/12, un timeout e zero fallimenti. Video frontale completo 1/1 riuscito, 41,24 s, con SHA256 verificati. Il candidato resta sperimentale e non sostituisce checkpoint13. Prima di altri training va diagnosticato il timeout.


### BC+SAC V5 — diagnosi e recupero anticipato della presa

V4 perde l’episodio11 L4→R3: presa verificata, peg sollevato, ma pressione PSM1 in calo prima dei2s di stabilità. Il precedente recupero richiedeva già il flag lift e non interveniva. Aggiunta opzione esplicita `recovery_before_lift`: richiede presa misurata, fondo peg≥43mm, holder PSM1, pinze PSM2 senza contatto e pressione sotto0,25N. Nessun flag, goal o fase è assegnato; stessi limiti di forza/coppia e criteri fisici.12 controlli CUDA superati. Avviata variante con identici pesi SAC/BC su12 reset originali; video obbligatorio. Solo dopo12/12 partirà un nuovo training V5 con baseline, warm replay, confronto12, video e seed indipendente62002. Pesi/configurazioni originali V4 e checkpoint13 integri. Stato: `logs/grasp_v2/bc_sac_recovery_retry_v5/status.json`.


V5 conclusa e verificata il14 ottobre: il recupero anticipato risolve il precedente episodio fallito (12/12 sugli stessi reset). Nuovo training BC+SAC12000 transizioni concluso; confronto con baseline12/12 superato. Sul seed indipendente62002:11/12, un timeout dell’episodio8 L4→R3 dopo grasp ma prima del lift stabile, zero fallimenti fisici. Nessuna promozione: checkpoint13 resta il riferimento. Prima di attribuire il limite a SAC serve il confronto BC a residuo zero sullo stesso seed indipendente. Video candidato `logs/grasp_v2/videos/bc_sac_residual_v5.mp4`:1/1 riuscito,42s,1050frame1280×720, intera decodifica e SHA256 verificati, visuale frontale controllata. Nessun training o valutazione residua attivo.


### 16 ottobre — nuovo piano con dimostrazioni dVRK reali

Decisione utente: preservare checkpoint13 come riferimento; registrare dimostrazioni reali, costruire IL visiva e reward, poi variante safety-tuned e confronto. Setup confermato: ROS1 precedente, endoscopio monoculare su topic, nessun sensore di forza alle pinze; segnali motori da verificare. Definito protocollo passivo rosbag, sincronizzazione/frame, conversione offline, replay cinematico distinto da replay fisico e test separati. Nessun nuovo training avviato. Primo risultato richiesto: inventario topic/versione e un bag pilota. Documento: `docs/DVRK_REAL_TO_ISAAC_PLAN.md`.


### 16 ottobre — cilindri board da12mm

Richiesta utente applicata alla scena corrente: altezza cilindri25→12mm, base appoggiata sulla board aZ10mm, centroZ16mm e puntaZ22mm. Aggiornate le quote delle collisioni geometriche SAC. Verifica Isaac/CUDA sui12 cilindri USD e3 passi di fisica; tutti55 file checkpoint13 SHA-verificati integri. Audit: `logs/scene_audits/posts_12mm_20261016_102947_526048.json`. Nessun training o nuovo checkpoint.
