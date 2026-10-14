# Curriculum autonomo — lavoro in corso

## Stato attuale — checkpoint13, verifica completata il 12 ottobre

I tre trasferimenti **L5→R2, L6→R1, L4→R3** usano gli stessi pesi BC di checkpoint12, con spostamenti liberi fino a100mm/s e trasporto stabile fino a24mm/s. Presa, scambio e inserimento restano lenti; l'apertura delle pinze vuote viene accelerata soltanto lontano dal peg. Nessun nuovo training. Aiuti alla presa e al deposito ancora attivi.

Risultati GPU: pilot **3/3**, conferma **12/12**, video sequenziale **3/3**. La conferma mantiene tutti i successi sugli stessi reset checkpoint12 e riduce la durata simulata media da **53,05 a46,30s** (−12,7%). Non è il massimo meccanico certificato: i profili superiori provati non superano la conferma fisica. Non è una verifica del full senza vincoli né della randomizzazione ampia.

```bash
conda activate PegTransfer
bash peg_grasp_v2.sh checkpoint13-play
```

Video: `logs/grasp_v2/videos/checkpoint13_three_episodes.mp4`; primo episodio: `logs/grasp_v2/videos/checkpoint13.mp4`. Sorgenti, configurazione, pesi e report archiviati in55 file SHA256 verificati. Video interamente decodificati; partenze, depositi e vista frontale controllati. [Report](milestones/checkpoint13-three-posts-fast-assisted.json), [download SSH](README.md#checkpoint13-spostamenti-piu-rapidi-interazioni-lente). Checkpoint10/11/12 verificati integri. Le sezioni seguenti sono storiche.

## Stato attuale — checkpoint12, 9 ottobre

Tre episodi consecutivi riusciti: **L5→R2, L6→R1, L4→R3**. Conferma aggiuntiva: **12/12**, con offset ±0,1 mm e yaw ±0,5° sulle stesse tre coppie.

Abbiamo conservato la rete precedente e allenato soltanto una piccola scelta neurale fra chiusura e lift, usando 12 sollevamenti fisici riusciti. BC sulla RTX 6000: 2.000 aggiornamenti, learning rate 0,0003, batch 1.024, rete 8–32–32–2. Tutti i 98 tensori originali sono invariati. Nessun insegnante durante il replay; restano attivi molla di presa, aiuto al mantenimento verticale, scarico della pinza sotto carico e guida di deposito.

```bash
conda activate PegTransfer
bash peg_grasp_v2.sh checkpoint12-play
```

Il comando esegue i tre percorsi in ordine con il modello conservato. Video: `logs/grasp_v2/videos/checkpoint12_three_episodes.mp4`. [Report e archivio](milestones/checkpoint12-three-posts-assisted.json), [download SSH](README.md#checkpoint12-tre-picking-e-tre-placement-con-presa-assistita). I paragrafi seguenti documentano gli esperimenti precedenti.

L’obiettivo è che la rete decida i movimenti dei PSM e l’apertura delle pinze. Ora aggiungiamo un aiuto temporaneo alla presa, come richiesto, e lo riduciamo progressivamente. I successi assistiti restano distinti dai successi senza vincolo.

## Stato attuale — 8 ottobre

**V55 interrotto; V56 riavviato.** La raccolta V55 conclude16 dei32 episodi richiesti, con12 task full riusciti salvati. Poi tenta di passare all'insegnante prima di un lift stabile e si ferma: nessun training avvenuto in V55. V56 riusa soltanto quelle12 traiettorie complete e verificate (10 TRAIN,2 VALID), lasciando esplicito che la raccolta originale è incompleta. Il raccoglitore ora attende il lift reale e ignora gli ambienti che hanno finito; le etichette non eseguite vengono escluse. Cinque regressioni CUDA superate, incluse protezione dei pesi e verifica dei dati. Training della sola pinza PSM2:2000 aggiornamenti, lr2e−5, batch1024, regolarizzazione0,1. Seguono le prove fisiche descritte sotto; nessuna nuova policy ancora certificata.

```bash
watch -n 5 cat logs/grasp_v2/autonomy_handover_refine_v56/status.json
tail -f logs/grasp_v2/autonomy_handover_refine_v56_startup.log
```

**V53 concluso:±5° verificato,31/32 full assistiti** (15/16 seed2468,16/16 seed54321), con offset±0,5 mm. Stessi pesi BC e codice originale checkpoint10; nessun nuovo training. Un episodio supera1 N durante il rilascio PSM1. Conservato **checkpoint11**,40 file verificati: pesi, sorgenti, configurazione e report. Replay: `bash peg_grasp_v2.sh checkpoint11-play --episodes 3`.

**V54 concluso e fermato alla verifica iniziale:** 30/32 a offset ±0,5 mm / yaw ±5° (15/16 su entrambi i seed). Sul seed54321 perde un successo precedente, con forza di 1,016 N durante il rilascio PSM1. Le rotazioni e gli offset maggiori non sono stati provati. La copia del runtime modifica soltanto una riga di validazione degli argomenti; gli altri29 sorgenti e i pesi restano identici. La causa della differenza non è ancora stabilita. Prima di ampliare il dominio occorre correggere e riverificare il passaggio della presa. Checkpoint11 e la sua valutazione V53 rimangono conservati.

**8 ottobre — piano originale V55: correzione BC del passaggio della presa.** Il log del caso fallito mostra PSM2 che continua a chiudere a −0,02 rad/s con forze asimmetriche di 0,350 / 0,764 N, prima del superamento di1 N. Raccogliamo32 episodi su un nuovo seed55001, offset ±0,5 mm / yaw ±5°. La rete esegue i primi2100 passi; soltanto nella raccolta l'insegnante recupera da un lift realmente misurato, alleggerisce PSM2 sopra0,65 N e ricentra anche durante il rilascio PSM1. Nessuna modifica alle pose del peg o ai criteri di successo.

Solo episodi full riusciti entrano nel dataset, con episodi interi separati per TRAIN/VALID. Si affina esclusivamente la rete della pinza PSM2 nelle opzioni10/11: 2000 aggiornamenti CUDA, batch1024, lr0,00002 e regolarizzazione verso la rete originale0,1. Gli altri pesi e il runtime di inference rimangono identici. La correzione del ricentraggio dell'insegnante non viene trasferita automaticamente ai movimenti neurali: se le sole pinze non bastano, le verifiche fisiche fermeranno questa candidata.

Prima della promozione servono3/3 full fissi e almeno15/16 su ciascuno dei due seed2468/54321, sia a yaw±2° sia a yaw±5°, conservando ogni precedente successo confrontabile. Velocità2×, spring firm1 e guida del deposito restano attivi. Solo dopo tutti i confronti e un video frontale full riuscito verrà conservato checkpoint12 con `videos/checkpoint12.mp4`. Al primo confronto insufficiente si conserva checkpoint11 e si ferma il workflow. Non è SAC; non è ancora verificata una nuova policy.

```bash
cat logs/grasp_v2/autonomy_handover_refine_v55/status.json  # raccolta interrotta
```

```bash
watch -n 5 cat logs/grasp_v2/autonomy_isolated_domain_v54/status.json
```

**V52 concluso:16/16 full assistiti con checkpoint10 originale sul seed54321**, compreso il caso9 perso in V51. Report e chiusura verificati;52 file dell'archivio integri. La causa della differenza tra le prove rimane da stabilire: questo replay non cancella il precedente15/16 e non dimostra una regressione del runtime o della GPU.

**V53 avviato:** stesso codice e stessi pesi archiviati, stesso offset ±0,5 mm, velocità2× e assistenza; cambia soltanto la rotazione a **±5°**.16 episodi per seed2468/54321; servono almeno15/16 per ciascuno. Questa è una verifica della generalizzazione, non nuovo training. I contatti e i comandi neurali vengono registrati per il BC mirato se il nuovo livello fallisce.

```bash
watch -n 5 cat logs/grasp_v2/autonomy_archived_yaw_v53/status.json
```

**V51 concluso:30/32 sul livello iniziale ±0,5 mm/±2°.** Entrambi i seed fanno15/16; il seed54321 perde un caso prima riuscito. Il gate rigoroso si ferma: nessuna rotazione superiore o nuovo cilindro è stato provato. Tutti32 completano presa e lift PSM1; un caso cade durante chiusura PSM2, l'altro supera1 N durante il rilascio del donatore. Pesi/attore/controllo motore/vincoli/physics sono identici; i nuovi reset campionano esattamente le condizioni precedenti. Report e chiusure Isaac verificati,52 file checkpoint10 integri. [Riepilogo V51](logs/grasp_v2/autonomy_domain_v51/completion_summary.json).

**V52 avviato:** replay diagnostico di checkpoint10 con il suo codice originale sul seed54321. Registra decisioni neurali, movimenti e contatti per verificare se il caso si ripete anche nell'archivio. Nessun training o ampliamento della randomizzazione durante questa verifica.

```bash
watch -n 5 cat logs/grasp_v2/autonomy_baseline_audit_v52/status.json
```

**V51 avviato: più rotazione, più offset, poi più cilindri.** Prima verifica il comportamento già riuscito di checkpoint10, a pesi identici e velocità2×. Poi aumenta solo la rotazione (±5°, ±10°, ±20°, ±30°), mantenendo ±0,5 mm per asse; successivamente aumenta lo spostamento a ±0,75 / ±1 / ±1,1 mm. I reset ruotano attorno al centro del foro, senza muovere board o cilindri. Il limite geometrico conservativo del campionamento quadrato è circa1,18 mm per asse; valori maggiori vengono rifiutati prima della simulazione.

Infine introduce nuovi cilindri con variazioni piccole: prima due destinazioni, poi due sorgenti, quindi tutti i cilindri raggiungibili. Ogni coppia viene provata separatamente, con16 episodi per ciascuno dei due seed; servono almeno15/16 su entrambi. I cilindri **L1–L6 del codice appaiono a destra nella vista frontale** (picking PSM1); **R1–R6 a sinistra** (deposito PSM2). Le coordinate corrispondono alla scena fisica conservata. Raggiungibilità e contatti sui nuovi cilindri devono ancora essere verificati.

Il workflow si ferma al primo livello insufficiente e conserva i report per il successivo BC/DAgger mirato. **È una verifica della generalizzazione, non nuovo training:** evita di riaddestrare ciò che già funziona. Durante questa fase molla di presa, guida al deposito e velocità restano quelle di checkpoint10. Riduzione degli aiuti e fine-tuning SAC vengono dopo. Checkpoint10 resta sempre riproducibile.

```bash
watch -n 5 cat logs/grasp_v2/autonomy_domain_v51/status.json
tail -f logs/grasp_v2/autonomy_domain_v51_startup.log
```

**Controlli preparatori:**83 test CUDA passati, incluse geometria dei reset, corrispondenza dei12 cilindri, copertura di ogni coppia/seed e regressioni di attore/velocità. Archivio checkpoint10 verificato:52 file. Questi controlli non certificano ancora i nuovi livelli fisici.

**V43 è concluso: 14/16 full assistiti, chiusura Isaac verificata.** Tutti completano presa e lift PSM1. Caso1: la pinza PSM2 mantiene l'apertura con forze 0,119/0,175 N e 0,035 mm di errore dal proprio obiettivo; timeout senza handover. Caso15: scivolamento durante chiusura PSM2. [Analisi completa](logs/grasp_v2/autonomy_handover_diagnostic_v43/receiver_close_audit.json).

**V47/V48 sono conclusi:** recuperi supervisionati dal lift reale, poi BC della sola pinza PSM2 in chiusura/rilascio del donatore. V44 interrotto durante il prefisso: l'insegnante restava alla prima fase e non poteva assumere correttamente il controllo del peg già sollevato. V45 aveva corretto il passaggio ma usava la parete sbagliata rispetto al checkpoint: interrotto insieme al workflow V46 prima del training. V47 usa parete300°, profondità4 mm, tilt0°, apertura0,25 rad e corregge il passaggio esclusivamente nella raccolta e verifica prima il lift e i contatti PSM1. Nessun flag di successo viene impostato dal recupero.

V48 accetta solo episodi full riusciti, completi, assistiti e con ricevuta di chiusura verificata. Esclude le etichette precedenti al passaggio all'insegnante; TRAIN/VALID separati per episodio. Nuova testa neurale PSM2, 2.000 aggiornamenti CUDA, lr 5e−5, batch 1.024, supervisionata dalle azioni realmente registrate. Ogni tensore preesistente resta identico. Risultati fisici: pilot3/3; ±0,25 mm / ±1°:16/16 e15/16; ±0,5 mm / ±2°:15/16 e16/16 sui seed nuovi2468/54321. **Checkpoint9 conserva V48 e i suoi report, 47 file verificati tramite SHA256.** Non è SAC né full nativo senza assistenza.

**V49 è concluso e scartato:** baseline3/3; ciascuna velocità rapida0/3, presa PSM1 raggiunta ma lift non stabile, nove timeout totali. Il replay rifiutava quindi correttamente una selezione assente. Messaggio corretto per non mostrare un traceback.

**V50 è concluso e verificato:** scelto il profilo2×, libero fino a50 mm/s e trasporto stabile fino a6 mm/s, contatto3 mm/s. Mantiene rotazioni e velocità minima di approccio originali. Pilot3/3; randomizzazione ±0,5 mm / ±2°:15/16 e16/16,31/32 successi e ogni precedente caso riuscito conservato. Durata media simulata51,97→47,64 s (−8,3%). Livelli3×/4× scartati per casi persi. Pesi BC V48, pinze/progressi/vincoli identici; nessun nuovo training. **78 controlli CUDA passati e dieci report fisici con chiusura verificati.**

Conservato **checkpoint10**,52 file con pesi, codice e configurazione2× congelati più report e riferimenti. Replay `bash peg_grasp_v2.sh checkpoint10-play --episodes 3`. Il tempo reale può dipendere da rendering e carico della workstation; il miglioramento riportato è sul tempo simulato. [Milestone](milestones/checkpoint10-assisted-full-bc-transit.json).

```bash
watch -n 5 cat logs/grasp_v2/autonomy_transit_speed_v50/status.json
# riferimento conservato, dal desktop
bash peg_grasp_v2.sh checkpoint9-play --episodes 3
# velocità maggiore verificata, codice e configurazione conservati
bash peg_grasp_v2.sh checkpoint10-play --episodes 3
```

V41 ha concluso il curriculum: **31/32 full assistiti al 100%**, **29/32 al 50%**. La riduzione si è fermata. Checkpoint8 e i suoi sorgenti restano immutati.

**V42 è concluso sulla RTX 6000:** 31/32 successi assistiti a ±0,25 mm / ±1°; 30/32 a ±0,5 mm / ±2° (16/16 e 14/16). Il secondo livello non supera la soglia di 15/16 per ciascun seed. I livelli maggiori non sono stati eseguiti. Il piolo L5 e la destinazione R2 rimangono fissi; l'assistenza resta completa.

**Velocità originale conservata:** 1,5× in aria / 1,25× in contatto conclude 13/16; 1,5× solo in aria conclude 11/16. Entrambi perdono successi del riferimento e sono respinti. Non sono cambiati giunti, guadagni, chiusura delle pinze o tempi delle verifiche fisiche.

I due errori del secondo livello avvengono nella chiusura di PSM2: un timeout e una caduta. PSM1 completa presa e lift in tutti i 32 casi. Perciò non è stato eseguito il fine-tuning dell'approccio PSM1. **V43 ripete i 16 casi del seed 9876 con registrazione delle osservazioni, dei comandi e degli obiettivi neurali:** serve a distinguere arresto della chiusura appresa, errore di posa e scivolamento. Nessun nuovo training o aumento dei vincoli viene dichiarato. **70 controlli CUDA V42 superati.**

```bash
tail -f logs/grasp_v2/autonomy_handover_diagnostic_v43_startup.log
cat logs/grasp_v2/autonomy_handover_diagnostic_v43/diagnostic_status.json
# Ultimo modello/velocità verificati; --randomized usa ±0,25 mm / ±1°
bash peg_grasp_v2.sh autonomy-motion-play
bash peg_grasp_v2.sh autonomy-motion-play --randomized
```

**V38: 3/3 task completi senza assistenza nella prova fissa L5 → R2.** La robustezza con posizioni variate non è ancora certificata.

**V39: stessi pesi, 3/3 task completi con assistenza alla presa.** Una molla virtuale limita lo scivolamento dopo che i sensori rilevano una presa. Applica al massimo 0,15 N e 0,002 Nm; non imposta la posizione o velocità del peg. Nel pilot viene rilasciata prima del deposito. La rete continua a scegliere movimenti e comportamenti; non riceve le fasi dell’insegnante.

È ancora **Behavior Cloning (BC), non SAC**. La correzione del contatto V38 è stata allenata sulla RTX 6000: 6.000 aggiornamenti, lr 0,0001, batch 1.024, 15 episodi TRAIN e quattro VALID. In questo primo curriculum i pesi restano fissi: misuriamo quanta assistenza serve prima di decidere un nuovo training. **64 controlli CUDA superati.**

Riduzione prevista: **100% → 50% → 25% → 10% → 0%**. Ogni livello richiede almeno 15/16 trasferimenti completi su ciascuno dei seed 123 e 9876, con variazioni di ±0,25 mm / ±1° intorno a L5 → R2. Se un livello fallisce, la procedura si ferma: useremo i suoi errori per il training successivo. Solo il livello 0% può certificare il full senza vincolo. Le tolleranze di deposito e il limite di forza delle pinze restano invariati.

La conferma V39 con piccole variazioni conclude **13/16 full assistiti**: una caduta, un arresto per forza e un timeout di deposito. Il curriculum V39 si ferma senza ridurre l’assistenza. V41 prova separatamente una presa assistita più salda (0,4 N; 0,004 Nm) e una guida di centraggio (massimo 0,15 N; 0,002 Nm), attiva solo dopo il rilascio vicino alla destinazione. Anche la guida deve spegnersi prima del secondo di stabilità richiesto per il successo. **64 controlli CUDA superati.** Gli stessi pesi BC sono conservati. **Il pilot V41 è concluso: 3/3 full assistiti**, chiusura corretta e vincoli spenti al deposito. È conservato come checkpoint8; la conferma conclude 31/32 al 100% e 29/32 al 50%, dove il curriculum si arresta. V40 aveva completato 3/3 task ma fallito la chiusura per un buffer PyTorch: V41 corregge quel problema, con regressione CUDA verificata. Il curriculum richiede anche la chiusura corretta del simulatore, oltre al report fisico.

Report del curriculum di riduzione concluso:

```bash
cat logs/grasp_v2/autonomy_full_constraints_curriculum_v41/status.json
cat logs/grasp_v2/autonomy_resume_status.json
```

Il pilot V41 è superato; il curriculum ha verificato 16 episodi per seed al 100% e50%. Arresto al primo livello non superato. La precedente conferma V39 è in `logs/grasp_v2/autonomy_full_assistance_curriculum_v39/status.json`.

Checkpoint5 e checkpoint6: tutti i 27 file di ciascun archivio ricontrollati e intatti. Il peg attuale usa contatti cedevoli, ma non è una gomma deformabile calibrata per sim-to-real.

Il nuovo **checkpoint7** conserva il pilot full assistito 3/3 e i suoi sorgenti. Per rivederlo sul desktop: `bash peg_grasp_v2.sh checkpoint7-play`; per registrarlo da SSH: `bash peg_grasp_v2.sh checkpoint7-video`. Il report specifica che non è una certificazione nativa o una prova di robustezza.

Il riferimento assistito più recente è **checkpoint10 (V50)**, stessi pesi V48 e traslazioni2× verificate su ±0,5 mm / ±2°: `bash peg_grasp_v2.sh checkpoint10-play --episodes 3` / `checkpoint10-video`. Checkpoint9 conserva la velocità originale. Checkpoint8, checkpoint7 e il suo video verificato restano conservati. [Report checkpoint10](milestones/checkpoint10-assisted-full-bc-transit.json).

## Risultati verificati

- **48/48 pickup dell’insegnante fisico**, con posizione variata di ±0,5 mm e rotazione di ±2° intorno al post L5. Servono come esempi; non sono successi della rete.
- Le prime reti BC, anche con memoria, hanno fallito le prove autonome. Una loss bassa non bastava.
- Una rete con scelta appresa della fase ha ottenuto **4/4 prese reali**, ma **0/4 lift**: la forza cresceva troppo durante il sollevamento. Il modello è conservato in `autonomy_linear_options_v2`.
- È stato corretto un errore nel comando delle pose: il limite sulle singole componenti cambiava la direzione dei movimenti lunghi. Le prove della correzione usano copie separate degli stessi pesi.
- La prima raccolta di recuperi dà **13/16 pickup**, con insegnante attivo: aggiunge 26.978 transizioni al dataset.
- La variante con memoria della posa ha ottenuto **3 pickup riusciti tra i 4 episodi iniziali**. Il vecchio conteggio parallelo poteva contare ripetizioni degli ambienti più veloci: il report di quella prova non viene usato per promuovere il modello. La nuova valutazione v8 completa ogni episodio assegnato una volta: **21/32 pickup autonomi**, 32 prese, 11 timeout, nessun arresto per forza. Non supera ancora il criterio del 90%. [Report rigoroso](logs/grasp_v2/autonomy_curriculum_v4/eval_round0/evaluation.json).
- La verifica del controllo ripristinato dà **4/4 pickup dell’insegnante**. La riapertura automatica per limite di forza non ha migliorato il lift ed è stata rimossa dalle nuove prove; la sua variante resta conservata.
- La prima prova wall→wall dà **0/2**. Wall→vertex ottiene una presa iniziale PSM2, ma la prova completa rivela che il peg scivola subito dopo il rilascio. Il vecchio risultato 1/1 riguarda soltanto quel passaggio momentaneo: **non certifica una tenuta stabile**. [Audit](logs/grasp_v2/autonomy_handover_vertex_v5/handover_hold_audit.json). Le nuove prove richiedono due secondi di sostegno del solo PSM2, sopra 37,5 mm, con contatti opposti stabili. È in prova un rilascio graduale di PSM1 e una presa PSM2 più salda. Checkpoint4 conserva il task completo riuscito con il suo vincolo temporaneo.
- **62/64 pickup autonomi riusciti con la variante v11**, senza insegnante né vincolo, su due seed. La prima prova separata aveva dato 16/16 e forza massima 0,43 N; nelle conferme c’è un arresto per forza per gruppo. La pinza usa una rete che decide chiusura, tenuta e apertura dai contatti. [Prima prova fisica](logs/grasp_v2/autonomy_pressure_bc_v11_eval16/evaluation.json).

## Strategia

1. Verificare presa e sollevamento PSM1 da soli.
2. Aggiungere esempi di recupero dagli stati visitati dalla rete (DAgger).
3. Richiedere almeno il 90% di pickup su due valutazioni da 32 episodi con seed diversi.
4. Estendere gradualmente le posizioni e verificare handover e placement fisici prima di allenarli.
5. Usare SAC per migliorare un comportamento già funzionante: velocità, forza e robustezza.

Questo criterio verifica la configurazione provata; non dimostra che il modello funzioni in tutte le situazioni.

## Checkpoint5 conservato

Il curriculum v5 è terminato: **31/32 con seed 123 e 31/32 con seed 9876**, quindi 62/64 pickup riusciti. Ciascun gruppo ha un arresto per forza eccessiva e nessun timeout. È superato il criterio del 90% per il pickup; il task completo resta aperto.

```bash
conda activate PegTransfer
# GUI sul desktop della workstation
bash peg_grasp_v2.sh checkpoint5-play
# Video frontale, anche da SSH
bash peg_grasp_v2.sh checkpoint5-video
```

Ogni video va in una cartella nuova `logs/grasp_v2/checkpoint5-video_.../inference_front.mp4`. Il modello, i sorgenti verificati e i report sono in `.checkpoint_runs/checkpoint5-native-pickup-bc`. [Manifest del checkpoint](milestones/checkpoint5-native-pickup-bc.json).

Per seguire l'esperimento separato sul trasferimento fisico:

```bash
tail -f logs/grasp_v2/autonomy_full_wall_fixed_donor_v13_startup.log
```

Il curriculum v4 è terminato senza promozione: la nuova imitazione non ha migliorato il modello. V11 mantiene i pesi di movimento già funzionanti e addestra soltanto il controllo della pinza.

## Parametri essenziali

- Simulazioni e training sulla RTX 6000 Ada, dispositivo `cuda:0`.
- BC con encoder 256+256 e scelta neurale tra comportamenti appresi. Le pose dipendono dalla geometria osservata; il comando delle pinze è appreso. Nessuna fase dell’insegnante è fornita alla rete durante l’esecuzione. Il candidato v8 usa memoria delle transizioni osservate negli episodi di training e conserva la posa scelta dalla rete durante ciascun comportamento. Le pinze restano controllate dalla rete continuamente. Non è una politica priva di vincoli decisionali.
- 98 ingressi fisici, con mascheramento delle caratteristiche inutili nel pickup; 14 azioni per movimento, rotazione e pinze dei due PSM. IK e limiti generali di velocità traducono le azioni in comandi articolari.
- Batch 1024, learning rate 0,0001, 12.000 aggiornamenti nel curriculum attuale. Gli episodi interi di validazione sono esclusi dal training.
- Controllo a 100 Hz, fisica a 400 Hz. Velocità massima 25 mm/s senza contatto e 3 mm/s con contatto; chiusura delle pinze rallentata a 0,02 rad/s con contatto. La variante del limite di forza a 0,40 N non ha superato la prova ed è conservata come esperimento fallito.
- Presa misurata tramite contatti opposti e tenuta; lift mantenuto per 2 s. Attrito 0,9, massa del peg 5 g. Il peg è rigido con contatti elastici, non un corpo deformabile.

Ogni modello salva pesi, sorgenti e hash. L’inference verifica la copia originale, così le modifiche successive non cambiano i vecchi checkpoint. Per ora il task usa lo stato della simulazione e la presa wall; visione e scelta tra prese diverse restano da sviluppare.


Le raccolte nuove salvano anche `completed_demonstrations.npz`: contiene soltanto episodi riusciti e completi già terminati. Se il processo si interrompe, questi esempi restano disponibili; la raccolta complessiva non viene dichiarata completa.

## Correzione della pressione — pickup verificato

La variante `autonomy_jaw_feedback_v9` imita piccoli cambiamenti dell’apertura: **1/16 pickup**, con 15 arresti per forza, e non viene promossa. V10 mantiene la presa ma resta nella chiusura: diagnosi interrotta, senza dichiarare episodi completi. V11 usa una fascia di tenuta più consistente nelle etichette offline, 0,36–0,40 N: **16/16 pickup** nella prima valutazione e **62/64** nelle due conferme. La scelta fra chiudere, tenere e aprire viene fatta dalla rete usando i contatti; quelle soglie non sono un controller applicato durante l’inference.

Il training aggiunge stati di calibrazione con forze modificate: servono a insegnare la risposta fra contatto debole e pressione eccessiva, **non sono dimostrazioni fisiche riuscite**. I pesi di movimento e scelta del comportamento restano fissi. Nuova rete della pinza: 64+64, learning rate 0,0003, 6.000 aggiornamenti, batch 1024. Gli episodi di validazione restano separati. 25 controlli CUDA superati, compreso il nuovo criterio di handover; i controlli software non sostituiscono le prove fisiche.

```bash
cat logs/grasp_v2/autonomy_curriculum_v5/status.json
tail -f logs/grasp_v2/autonomy_full_wall_fixed_donor_v13_startup.log
```

Le prove full v7 e v8 aumentano la pressione PSM2 e rallentano il rilascio di PSM1. V8 segue anche i piccoli spostamenti misurati del peg durante l'apertura del donatore. È ancora osservato scivolamento: queste prove non certificano il trasferimento e non promuovono un checkpoint completo.

## Estensione delle posizioni — in corso

Raccolta fisica su L5/L6, ±0,8 mm e ±4°, con 16 ambienti CUDA. Il curriculum prova il checkpoint5 nelle nuove condizioni; se non passa, raccoglie recuperi e allena un nuovo candidato BC con controllo neurale della pressione. Nessun risultato nuovo viene promosso prima di due verifiche da 32 episodi, con almeno il 90% anche su ogni post e almeno otto casi per post. La posizione di placement non è ancora certificata dal solo pickup.

```bash
tail -f logs/grasp_v2/autonomy_positions_curriculum_v1_startup.log
cat logs/grasp_v2/autonomy_positions_curriculum_v1/status.json
```

La prova separata del task completo usa una presa wall PSM2 a 300°. È stato corretto il blocco della chiusura al primo contatto debole; una caduta sulla board prima dell'handover ora termina il tentativo come fallimento. Le prove v8/v9 interrotte restano archiviate, senza conteggiarle come episodi completati. **28 controlli CUDA superati**.

La raccolta L5/L6 è completa: 29/32 successi dell'insegnante, 3 timeout, 68.705 transizioni valide. Il curriculum valuta ora la rete. V10 non ha risolto il trasferimento; V11 corregge l'allineamento PSM2 seguendo piccoli cambiamenti della parete misurata durante il contatto. È ancora un esperimento fisico, non un nuovo checkpoint full riuscito.


## Ultima ripresa — 6 ottobre

Il **checkpoint5** resta il pickup autonomo verificato: **62/64**, L5 ±0,5 mm / ±2°. Il **checkpoint4** conserva la sequenza completa assistita. Il **checkpoint6** conserva 3/3 full fisici dell'insegnante, senza aggancio al peg: **la rete full non è ancora verificata**. Verificati i 27 file di checkpoint5 e checkpoint6.

La raccolta full ampliata ha già salvato 10 episodi completi riusciti (62.592 transizioni). Sono usati insieme ai tre episodi del checkpoint6, con episodi separati per la validazione. La raccolta termina **10/16**, una caduta e cinque timeout al placement; nessuna promozione della rete. Il replay video dell'insegnante ha mostrato un placement fuori tolleranza ed è conservato come diagnosi, non come successo.

Le candidate full precedenti si bloccano o perdono il peg. V15 si ferma al primo waypoint: le etichette degli ultimi campioni descrivono ancora la fase precedente. **V17** prova l'anticipazione di 20 campioni già usata nel pickup riuscito. La rete sceglie fasi, pose e pinze; in inference riceve stato fisico, non l'insegnante. Training su RTX 6000: 24.000 aggiornamenti BC, batch 1.024, lr 0,0001; poi controllo neurale delle pinze. 36 verifiche CUDA superate.

In parallelo, V16 corregge il placement dell'insegnante misurando nuovamente la presa PSM2 dopo il rilascio di PSM1. Il peg rimane libero. Le prove sulle posizioni L5/L6 più ampie hanno ottenuto 5/32 e 13/32: nessuna promozione.

```bash
tail -f logs/grasp_v2/autonomy_full_native_anticipation_bc_v17_startup.log
tail -f logs/grasp_v2/autonomy_full_native_loaded_transform_v16_startup.log
cat logs/grasp_v2/autonomy_resume_status.json
```

Prima completiamo e verifichiamo il BC full nativo. Poi allarghiamo picking e placement e facciamo fine tuning SAC. Il precedente SAC libero resta 0/16: non viene presentato come riuscito.

**Esito aggiornato:** il placement V16 conclude **12/16**, una caduta e tre timeout (prima 10/16). V17/V18 ottengono prese valide ma restano prima del lift: diagnosi interrotte, nessuna promozione. DAgger esegue 12 s continui della rete, poi lascia il recupero all'insegnante, senza alternare le pinze a ogni passo. V20 termina **4/6 full**, con due cadute durante l'handover; tutti e sei hanno superato il lift.

Il workflow **V21** usa 19 episodi fisici full riusciti e termina **0/3** nella verifica della rete: raggiunge la presa ma non passa alla chiusura. Nessuna promozione. **V22** allena una nuova rete di scelta: riceve il comportamento scelto prima, l'errore rispetto alla propria posa appresa e le misure fisiche. Le transizioni sono insegnate soltanto dopo che avvengono nelle dimostrazioni; non usa un timer o l'insegnante durante l'esecuzione. Movimento e pinze restano identici a V21. Training CUDA: 10.000 aggiornamenti, lr 0,0001, batch 1.024; quattro episodi interi esclusi dal training. **40 controlli CUDA superati; verifica fisica in corso.**

```bash
tail -f logs/grasp_v2/autonomy_full_pressure_calibrated_router_v26_eval3_startup.log
cat logs/grasp_v2/autonomy_resume_status.json
```

V22 resta al primo waypoint. **V23 supera il blocco: tre prese native e due lift stabili**, ma la scelta resta ferma nella chiusura o al punto di transfer. Entrambe sono diagnosi interrotte con audit, non prove full completate. **V24** usa intensità e opposizione delle forze, apertura delle pinze, progresso fisico e distanza dal proprio obiettivo; elimina direzione assoluta delle forze e altezza assoluta dalle scorciatoie del selettore. Il movimento e le pinze rimangono fissi. **41 controlli CUDA superati.**

**V26 in verifica:** il selettore viene calibrato sulla pressione della rete delle pinze, usando le transizioni già osservate nelle dimostrazioni. La calibrazione offline non conta come successo fisico e non introduce un controller durante l'inference. Su due stati reali prima bloccati sceglie ora lift e reach PSM2; deve confermarli nella sequenza completa. **42 controlli CUDA superati.** I checkpoint5/6 restano integri. Per certificare un nuovo full servono prima 3/3 episodi e poi almeno 15/16 su ciascuno di due seed, stessi pesi e nessun insegnante o aggancio al peg.


### 6 ottobre — ripresa V28, presa conservata

V26 e V27 terminano entrambe **0/3 full nativi**: tre prese PSM1, due lift stabili, due cadute e un timeout. V27 conserva tra i comportamenti gli obiettivi che nelle dimostrazioni restano fermi, ma non risolve l’handover. Nessuna candidata viene promossa.

V28 incorpora i pesi originali del **checkpoint5 (pickup 62/64)** per approccio, presa e lift. La rete full continua a scegliere i comportamenti e controlla il resto del task. È una composizione di reti BC: non è SAC e non aggiunge un insegnante o un aggancio al peg. I pesi conservati sono verificati identici; 27 file di ciascuno dei checkpoint5/6 verificati. **45 controlli CUDA superati**. La prova di tre episodi full è avviata; il riuso dei pesi non certifica automaticamente il risultato della nuova composizione.

```bash
tail -f logs/grasp_v2/autonomy_full_preserved_pickup_v28_eval3_startup.log
cat logs/grasp_v2/autonomy_resume_status.json
```


V28 resta una diagnosi incompleta: tre prese, un lift stabile e una caduta conclusa; due scene restano ferme senza contatto. Interrotta con audit del processo. **V29** conserva anche il selettore neurale del pickup: riusare soltanto i decoder non aveva conservato quando la rete cambiava comportamento. La continuazione full è ancora scelta dalla rete full, senza soglie di contatto o timer di task aggiunti. **46 controlli CUDA superati**, verifica fisica avviata.


**V29 conclude 0/3 full, ma 3/3 prese e lift stabili.** Tutti cadono durante la chiusura PSM2. La posa appresa cambia di circa 2,5 mm e 0,15 rad all’ingresso in chiusura: un movimento indesiderato sotto carico. V30 usa la posa precedente nelle opzioni con aggiustamenti piccoli nelle dimostrazioni di training (tolleranze 2 mm / 0,13 rad). Non introduce pose dell’insegnante durante l’inference; pesi neurali invariati e criterio di successo invariato. È una semplificazione della memoria di posa, da verificare fisicamente; non un nuovo successo dichiarato.
