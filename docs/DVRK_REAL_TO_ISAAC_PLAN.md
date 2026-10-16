# Dal dVRK reale a una policy visiva in Isaac

Piano aggiornato il16 ottobre2026. Setup comunicato: ROS1 su installazione dVRK precedente, endoscopio monoculare già pubblicato su un topic, nessun sensore di forza dedicato sulle pinze; disponibilità dei segnali dei motori da verificare.

Questo documento pianifica registrazione e conversione. Non avvia registrazioni, movimenti reali, replay sul robot o training.

## Obiettivo e riferimenti

Conservare **checkpoint13** senza modifiche. Nuova linea indipendente:

registrazioni reali → dataset immagini/azioni → IL visiva vanilla → copia safety-tuned → confronto.

La reward appresa dalle dimostrazioni è un modulo separato: BC può partire senza reward. Per ricavare una reward informativa servono un'ipotesi di inverse RL oppure valutazioni/preferenze su esecuzioni e segmenti; una collezione di successi non determina una reward unica e non prova la safety.

Il confronto causale principale è fra la nuova policy visiva vanilla e una sua variante safety-tuned. Checkpoint13 è un riferimento storico con osservazioni privilegiate di stato e aiuti fisici: confronto utile, ma non isola il solo effetto del tuning. Se il tuning introduce anche RL, aggiungere una variante RL con lo stesso budget e la stessa reward di task, senza i nuovi costi di safety. Non chiamare "priva di safety" una baseline che ha già protezioni o preferenze di qualità.

## 1. Ricognizione del sistema ROS1

Eseguire sul computer dVRK, nel terminale ROS già configurato:

```bash
rosversion -d
rosversion dvrk_robot
rostopic list -v
```

`rosversion dvrk_robot` può non trovare il pacchetto in una particolare installazione: non è necessario aggiornare lo stack per procedere alla registrazione. I nomi veri si ricavano dai topic attivi e dai relativi tipi.

Per ogni topic selezionato, usare `rostopic info NOME`, `rostopic type NOME` e `rostopic hz NOME` (interrompere quest'ultimo conCtrl+C). Per la camera anche `rostopic bw NOME`. Controllare un header con `rostopic echo -n 1 NOME/header`, se il messaggio lo contiene. Evitare di stampare l'intera immagine in console.

| Segnale da conservare | Possibili nomi legacy, da verificare | Nomi CRTK più recenti, da verificare |
| --- | --- | --- |
| Posa misurata PSM1 ePSM2 | `/dvrk/PSM1/position_cartesian_current` | `/PSM1/measured_cp`, eventualmente `local/measured_cp` |
| Stato dei giunti | `/dvrk/PSM1/state_joint_current` | `/PSM1/measured_js` |
| Riferimento cartesiano del controller | `/dvrk/PSM1/position_cartesian_desired` | `/PSM1/setpoint_cp` |
| Riferimenti dei giunti | `/dvrk/PSM1/state_joint_desired` | `/PSM1/setpoint_js` |
| Apertura/stato pinza, misurato e desiderato | eventuali `state_jaw_current` e `state_jaw_desired` | eventuali `jaw/measured_js` e `jaw/setpoint_js` |
| Camera | topic effettivo `Image` o `CompressedImage` | stesso requisito |

I nomi sopra sono esempi di famiglie API, non la configurazione già verificata del laboratorio. Registrare entrambi iPSM. Confermare il tipo reale: una posa può essere `PoseStamped` oppure `TransformStamped`; non assumere che il converter possa trattarli allo stesso modo.

Conservare inoltre, quando disponibili: `camera_info`, `/tf`, `/tf_static`, posa/stato ECM se muove la camera, eventi clutch/teleoperazione, warning/errori, stato operativo e `/rosout`. I comandi ROS diretti possono non apparire quando la teleoperazione gira internamente in cisst: i setpoint pubblicati sono importanti ma non vanno automaticamente chiamati intenzione dell'operatore.

PSM1 ePSM2 misurati, pinze e immagini sono prioritari. I MTM sono facoltativi per analizzare il gesto umano; per le azioni della policy servono soprattutto i movimenti eseguiti/richiesti sul lato PSM.

## 2. Registrazione passiva

Usare `rosbag record` con una lista esplicita dei topic verificati. Le librerie ROS1 restano sul computer di acquisizione o in un ambiente di conversione separato; Isaac non deve collegarsi in tempo reale a ROS1 per allenare sui dati.

Primo esperimento: un bag di circa60–90s con un solo trasferimento completo e pochi secondi prima/dopo. Deve contenere l'immagine originale con timestamp, tutti i segnali dei duePSM e le pinze. Un MP4 da solo non conserva questa associazione temporale. Se serve compressione, usare un'opzione supportata dalla versione locale e verificarne l'impatto; non assumere che aumentare il buffer risolva un disco insufficiente.

Esempio di struttura del comando, **da completare con i topic veri**:

```text
rosbag record -O pilot_001.bag TOPIC_CAMERA TOPIC_CAMERA_INFO TOPIC_PSM1_POSE ...
```

Avviare la registrazione prima dell'esecuzione manuale e terminarla conCtrl+C dopo il rilascio e la stabilizzazione del peg. Non usare una procedura di replay ROS sul sistema reale: un bag con topic di comando può ripubblicare movimenti. Il replay qui previsto avviene soltanto nel simulatore isolato.

Conservare il bag originale senza riscriverlo; aggiungere manifest con SHA256, distribuzione ROS, versione/configurazione dVRK, identificativo strumento, calibrazioni e sorgenti del converter. Prima di raccolte lunghe controllare con `rosbag info pilot_001.bag` frequenze/conteggi, immagini leggibili, perdite di messaggi e corretto completamento del file.

Il recorder deve restare passivo. Prima confrontare il carico e il tracking durante un pilot: niente elaborazioni visive pesanti sul processo di controllo dVRK. Conversione e training verranno eseguiti sulla workstation RTX6000.

## 3. Sincronizzazione e metadati degli episodi

Conservare sia il timestamp del messaggio sia quello di ricezione nel bag. Verificare se gli header sono validi, quale clock li genera e se l'immagine è datata all'acquisizione o alla pubblicazione. Essere nello stesso bag non elimina ritardi di camera/rete.

Allineare le pose al tempo di ciascuna immagine; interpolare posizioni e orientamenti suSE(3), con SLERP per i quaternioni. Per stati discreti usare una regola causale dichiarata. Non interpolare attraverso gap lunghi o confini di episodio; salvarne maschere di validità. Conservare separatamente misure, setpoint e gli eventuali comandi.

Registrare alla frequenza nativa utile dei publisher, senza aumentarla artificialmente. Ipotesi iniziale da verificare: immagini25–30Hz, cinematica almeno100Hz se disponibile. Frequenza della futura policy da scegliere dopo la misura delle latenze; il controllo servo resta separato. Immagini interpolate o duplicate non diventano nuove osservazioni indipendenti.

Per episodio salvare: operatore/sessione, sorgente e destinazione, posa iniziale del peg se misurabile, successo e stabilità finale, recuperi, scivolamenti, urti visibili, occlusioni, interruzioni, tempi delle fasi e motivi di esclusione. Distinguere sequenze corrette da sequenze problematiche senza eliminarne gli originali. Non provocare deliberatamente eventi pericolosi per raccogliere esempi negativi.

Iniziare con3–5 episodi per verificare il sistema; soltanto dopo il pilot pianificare30–50 dimostrazioni iniziali distribuite sulle tre coppie. Sono dimensioni di una prima raccolta, non una garanzia di training sufficiente. Separare train/validation/test per episodi interi e, dove possibile, per sessioni/operatori/configurazioni; mai mescolare frame adiacenti fra gli split.

## 4. Calibrazione reale–Isaac

La replica della scena è un buon punto di partenza, ma vanno verificati:

- intrinseci e distorsione dell'endoscopio monoculare; risoluzione, crop, zoom e eventuali flip;
- posa della camera rispetto alla board, compresa l'eventuale variazione dell'ECM;
- trasformazioni fra basi/RCM dei duePSM e board;
- TCP effettivo, lunghezza dell'utensile, orientamento, nome/segno/unità dei giunti;
- relazione fra apertura della pinza reale e pinza simulata;
- geometria e disposizione dei cilindri e del peg, massa e comportamento dei contatti da calibrare.

Salvare ogni posa con il proprio frame. Per esempio, se una posa è espressa nella base delPSM:

`T_board_tool = T_board_base × T_base_tool`.

Se il dVRK la esprime già rispetto all'endoscopio si usa invece la catena appropriata: non applicare due volte la trasformazione della base. PSM1 ePSM2 non condividono automaticamente la stessa origine. ROS usa spesso quaternioni `xyzw`; il nostro runtime usa `wxyz`. L'inserzione è una traslazione: non trattare tutti i giunti come angoli.

Con la monoculare non si ottiene la profondità metrica del peg sollevato dalla sola omografia del piano board. Per il pilot usare calibrazione, geometria nota e annotazioni/tracking validati; se il peg non è osservabile, dichiararlo. I timestamp delle occlusioni vanno conservati. Coordinate 3D ricostruite possono servire come annotazioni o riferimento di valutazione, non come input perfetti nascosti nell'actor visivo.

## 5. Conversione del bag e due tipi diversi di replay

Converter offline: legge i messaggi ROS1, verifica tipi/frame/unità, sincronizza, segmenta e produce un formato intermedio HDF5 con schema nostro versionato. HDF5 è il contenitore, non una conversione automatica verso qualsiasi task Isaac.

Struttura logica proposta per episodio:

```text
timestamp_image, timestamp_robot, validity
obs/rgb
robot/psm1/{pose_measured,pose_setpoint,joints,jaw}
robot/psm2/{pose_measured,pose_setpoint,joints,jaw}
actions/{delta_pose_psm1,delta_pose_psm2,jaw_targets}
annotations/{phase,outcome,event,quality,source,target}
calibration/{camera,frames,joint_map,jaw_map}
metadata/{source_bag_sha256,session,operator,split,schema}
```

**Replay cinematico:** segue giunti/pose registrati per controllare allineamento, TCP, pinze e immagine renderizzata. Muovere anche il peg sulla traiettoria osservata può servire a una ricostruzione visiva, ma non prova che il robot l'abbia preso o trasferito fisicamente.

**Replay fisico:** inizializza la scena e applica al robot i riferimenti adattati. Il peg si muove per contatti; misurare tracking, presa, scivolamento, urti e deposito. Documentare ogni aiuto. La stessa traiettoria reale può fallire in simulazione per differenze di calibrazione, contatti, attrito o attuazione. Non risolvere il disallineamento imponendo la posa dell'oggetto e dichiarando un successo.

Le azioni non sono semplicemente i giunti letti dal bag. Scegliere un unico contratto per l'actor: inizialmente delta di posa cartesiana e riferimento pinza per ciascunPSM, con frame, intervallo temporale e limiti espliciti. Usare setpoint/comandi quando la loro semantica e latenza sono verificate; altrimenti ricavare azioni da spostamenti misurati su un orizzonte dichiarato e validarli in replay. Non utilizzare osservazioni future come input della policy e non confondere velocità misurata con comando desiderato.

Non riutilizzare automaticamente scale di3mm/.15rad o azioni normalizzate del checkpoint13: dipendono dal suo intervallo di controllo e dai limiti del suo esecutore. Serve un adapter per il nostro task bimanuale e per i diversi riferimenti/jaw mapping. IsaacLab include registrazione e replay HDF5, ma non un importer universale ROSbag per questoPegTransfer.

Conservare dati reali e sintetici distinti. Un'immagine sintetica è un esempio valido solo se scena e azione sono coerenti: durante un replay fisico divergente non associare alla scena sintetica l'azione reale come se fosse ancora un esperto corretto.

## 6. IL visiva, reward e safety

**IL vanilla:** immagine monoculare e storia temporale → azioni dei duePSM. Le misure robotiche producono le etichette/il controllo di basso livello, non diventano automaticamente input dell'actor. Destinazione fissata nel primo pilot; per più destinazioni usare una specifica visiva esplicita. Un singolo frame può non bastare quando il peg o la pinza sono occlusi: confrontare un modello temporale/chunk di azioni, conservando la capacità di reagire al contatto.

**Reward:** prima BC, in parallelo definizione della reward. Opzioni: inverse RL sulle dimostrazioni oppure reward model da preferenze fra brevi segmenti e valutazioni esperte. Per iniziare propongo preferenze motivate e rubriche trasparenti; non serve lanciare subito AIRL. Distinguere progresso del task da costi di safety. Se una reward di preferenza incorpora già aspetti di safety, dichiararlo anche per la vanilla.

Una reward visiva imparata sulle sole immagini reali potrebbe riconoscere l'aspetto del reale invece della qualità del gesto. Verificarla separatamente su reale/simulato, fasi, operatori e scene non viste prima di usarla per RL. Valutazione finale e soglie di safety rimangono esterne alla reward ottimizzata.

**Forze:** senza sensori dedicati non etichettare le traiettorie come sicure rispetto alla forza di presa. Se esistono `JointState.effort` o `measured_cf`, registrarli come segnali/stime, non come misure validate del contatto. La documentazione dVRK descrive `measured_cf` come emulazione da correnti/coppie e Jacobiano, con importanti limiti del modello. Una cella sotto la board può misurare urti con la board, ma non tutta la compressione fra pinze e peg in aria. Per questa serve una misura o una stima specificamente validata.

Ora si possono valutare moto, precisione, scivolamento/cadute e collisioni visibili; ciò che è occluso o non misurabile va marcato come incerto. In Isaac le forze sono disponibili come riferimento simulato, senza trasformarle in certificazione reale. Tutte le policy vengono giudicate dagli stessi criteri di safety; successo e tempo restano secondari.

## Milestone e condizioni per proseguire

| Passo | Risultato concreto | Condizione per passare al successivo |
| --- | --- | --- |
| R0 | Inventario ROS1, topic, camera e calibrazioni | Segnali richiesti identificati; frequenze, frame e timestamp verificati |
| R1 | Primo bag e video sincronizzato | Un episodio intero leggibile, pinze presenti, qualità temporale accettabile |
| R2 | Dataset intermedio e replay cinematico | Sovrapposizione reale/simulato e mapping dei duePSM verificati |
| R3 | Replay fisico di3–5 episodi | Esiti e divergenze misurati; aiuti espliciti; nessun successo fittizio |
| R4 | Raccolta iniziale e IL visiva vanilla | Test separato, report di safety e video; checkpoint conservato |
| R5 | Reward verificata e variante safety-tuned | Confronto appaiato con vanilla, stesso budget/aiuti o ablation dichiarate |
| R6 | Confronto con checkpoint13 | Riferimento congelato e differenza fra osservazioni privilegiate e visive dichiarata |

Ogni nuovo checkpoint conservato avrà un video completo frontale dai propri pesi/sorgenti/configurazione, report e metadatiSHA256, come richiesto daAGENTS.md. Nessuna variante sostituisce la vanilla.

## Fonti tecniche

- [ROS1 rosbag: implementazione ufficiale del comando](https://github.com/ros/ros_comm/blob/noetic-devel/tools/rosbag/src/rosbag/rosbag_main.py).
- [dVRK API: pose misurate, setpoint, frame e stime di forza](https://dvrk.readthedocs.io/2.5.0/pages/development/api/arms.html). Documentazione moderna: non assume che il laboratorio abbia questa versione.
- [IsaacLab: teleoperazione, dataset HDF5 e replay](https://isaac-sim.github.io/IsaacLab/v2.0.1/source/overview/teleop_imitation.html). Verificata anche la presenza degli strumenti nella copia locale; il task richiede un adapter proprio.
- [Reward learning da preferenze](https://arxiv.org/abs/1706.03741).
- [Adversarial inverse RL e reward trasferibili](https://arxiv.org/abs/1710.11248).

Restano da ottenere: lista topic/versione ROS effettiva, camera_info/calibrazione, segnali effettivamente pubblicati per pinze e sforzi, un primo bag con metadata della scena.
