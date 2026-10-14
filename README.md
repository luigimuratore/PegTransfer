# Dual-PSM peg transfer

Progetto Isaac Sim / Isaac Lab per il peg transfer con due dVRK Patient Side Manipulators.

Le sei tracce di simulazione più grandi sono archiviate con Git LFS. Prima di clonare il repository, installare Git LFS ed eseguire `git lfs install`; per scaricare le tracce in un clone esistente, eseguire `git lfs pull` dalla cartella del progetto. I file mantengono nomi e contenuti originali.

**Ultimo risultato — checkpoint13:** spostamenti più rapidi, con presa, scambio e inserimento lenti. **12/12** successi nella conferma; durata media **46,3 s contro 53,0 s** di checkpoint12, circa **13% in meno**. Video con tre trasferimenti consecutivi riusciti: **L5→R2, L6→R1, L4→R3**. È BC con aiuti temporanei alla presa e al deposito. [Replay, video e report](#checkpoint13-spostamenti-piu-rapidi-interazioni-lente).

Report breve per la presentazione: [checkpoint1.txt](checkpoint1.txt), riferito al risultato tecnico checkpoint13. Nuova pipeline sperimentale: [BC + SAC residuale, parametri e comandi](docs/BC_SAC_RESIDUAL.md). Mantiene la BC congelata e gli aiuti fisici; i risultati SAC devono superare i confronti fisici prima di diventare un nuovo checkpoint.

Pilot BC+SAC V1 fermato prima degli aggiornamenti SAC: baseline **12/12**, raccolta BC iniziale **5/6**, con un timeout PSM2 su L4→R3. Nessun candidato o nuovo video prodotto. Corretto il supervisore, che aveva tentato la valutazione dopo l'arresto; ora verifica report, errori e artefatti oltre al codice d'uscita. Checkpoint13 rimane il riferimento conservato. [Stato V1](logs/grasp_v2/bc_sac_residual_v1/status.json).

**Rilancio V4:** stallo corretto con recupero temporaneo della pressione di PSM1; **6/6** trasferimenti completi sullo stesso seed prima fallito e **12/12** nella baseline variata, dieci controlli CUDA passati. Collisioni originali e criteri fisici invariati; aiuto alla molla e all'apertura solo nella perdita di presa prima del contatto PSM2. Seguire con `watch -n 5 cat logs/grasp_v2/bc_sac_residual_v4/status.json`. Il workflow esegue raccolta BC, SAC, conferma12 e registra il candidato in `logs/grasp_v2/videos/bc_sac_residual_v4.mp4`; il video sarà disponibile dopo queste fasi. Download dal Mac: `scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/bc_sac_residual_v4.mp4 ~/Downloads/`. [Configurazione e comandi](docs/BC_SAC_RESIDUAL.md).

Il checkpoint1 completa il trasferimento **L5 → R2**: 100/100 successi nella configurazione fissa con presa assistita. È stato addestrato per imitazione (Behavior Cloning); la pipeline SAC è disponibile per gli aggiornamenti tramite reward. Sequenziamento e controllo del braccio inattivo restano programmati. Il checkpoint2 verifica la presa fisica PSM1 senza agganci: 3/3 episodi riusciti. Il nuovo riferimento checkpoint3 completa la sequenza v2 **3/3 volte con un vincolo temporaneo sul peg**; usa la rete PSM1 già salvata e un controller programmato per il resto del task.

Il checkpoint4 estende il controllo neurale all'intera sequenza v2: **3/3 episodi completi verificati sulla RTX 6000**, con il vincolo temporaneo e la sequenza delle fasi programmata. I checkpoint precedenti restano conservati.

### Autonomia: BC, recupero dagli errori e posizioni variabili

La rete **V38 completa 3/3 trasferimenti nativi L5 → R2 nella prova fissa**. Non è ancora certificata con posizioni variate. V39 usa gli stessi pesi e aggiunge un vincolo temporaneo di presa: **3/3 trasferimenti assistiti**, senza insegnante durante l’inference. È BC, non SAC. La prima conferma assistita mostra ancora errori di presa e deposito. V41 prova una presa più salda e una guida temporanea di centraggio dopo il rilascio; il curriculum riduce i vincoli al 100%, 50%, 25%, 10% e 0%, soltanto con almeno 15/16 successi su ciascuno di due seed. Il checkpoint5 conserva il pickup nativo 62/64; checkpoint4 conserva il precedente full assistito. [Stato e comandi aggiornati](PEG_AUTONOMY.md).

Il primo curriculum varia la posizione del peg di ±0,5 mm e la rotazione di ±2° intorno al post L5. Sono in confronto BC con memoria, pose future e pose scelte direttamente dalla rete; DAgger raccoglie correzioni durante l'approccio. Una prova separata verifica l'handover fisico senza vincolo prima di addestrarlo. I modelli conservano pesi e sorgenti; `autonomy-play` verifica e usa la copia originale. [Procedura e comandi](PEG_AUTONOMY.md).

## checkpoint5: autonomous pickup, full transfer in progress

PSM1 raggiunge il peg, chiude la pinza e lo solleva con presa fisica: **31/32 episodi con seed 123 e 31/32 con seed 9876**. Posizione variata di ±0,5 mm e rotazione di ±2° intorno a L5. Un arresto per forza eccessiva in ciascun gruppo; nessun timeout.

È una rete **BC**, con scelta appresa dei comportamenti e controllo neurale della pinza. Usa lo stato della simulazione e IK; non usa un insegnante o agganci al peg durante l'inference. Il checkpoint5 certifica soltanto il pickup; i nuovi esperimenti full hanno cartelle separate. Pesi, codice e report sono conservati separatamente: le nuove modifiche non li sostituiscono. [Report del checkpoint](milestones/checkpoint5-native-pickup-bc.json).

Sul desktop della workstation:

```bash
conda activate PegTransfer
bash peg_grasp_v2.sh checkpoint5-play
```

Da SSH, per registrare un episodio con camera frontale:

```bash
conda activate PegTransfer
bash peg_grasp_v2.sh checkpoint5-video
```

Il video si trova nella nuova cartella `logs/grasp_v2/checkpoint5-video_.../inference_front.mp4`. I comandi verificano pesi e codice originali; conservare anche l'ambiente Isaac e gli asset locali sulla workstation.

## checkpoint7: full task with temporary grasp support

La rete BC completa **3/3 episodi fissi L5 → R2** con una molla virtuale che aiuta la presa. Il deposito avviene con il vincolo rilasciato. È un riferimento assistito: la robustezza con posizioni variate e il full senza vincolo non sono certificati da questa prova. Pesi, codice e report sono conservati separatamente; le nuove modifiche non li sostituiscono. [Report](milestones/checkpoint7-assisted-full-bc-pilot.json).

Sul desktop della workstation:

```bash
conda activate PegTransfer
bash peg_grasp_v2.sh checkpoint7-play
```

Da SSH, per registrare un episodio con camera frontale:

```bash
conda activate PegTransfer
bash peg_grasp_v2.sh checkpoint7-video
```

Il video viene salvato nella nuova cartella `logs/grasp_v2/checkpoint7-video_.../inference_front.mp4`. Questi comandi verificano l’archivio e ripristinano l’assistenza della prova originale. Richiedono l’ambiente Isaac e gli asset originali della workstation. Il [video già verificato](.checkpoint_runs/checkpoint7-assisted-full-bc-pilot-video/inference_front.mp4) mostra un episodio completo (49 s, camera frontale); [report](milestones/checkpoint7-assisted-full-bc-pilot-video.json).

## checkpoint8: full task with temporary grasp and placement constraints

La rete BC V41 completa **3/3 task L5 → R2** nella prova fissa. Una molla aiuta la presa; una guida temporanea centra il peg dopo il rilascio. Entrambe sono disattivate prima del successo. La chiusura del simulatore è verificata. Con ±0,25 mm / ±1° ha ottenuto **31/32 full assistiti** su due seed; dimezzando i vincoli, 29/32: la riduzione si è fermata. Questo checkpoint conserva il pilot assistito, senza certificare il full nativo. [Report conservato](milestones/checkpoint8-assisted-full-bc-constraints.json).

```bash
conda activate PegTransfer
# Inference sul desktop della workstation:
bash peg_grasp_v2.sh checkpoint8-play
# Oppure registra un nuovo video da SSH:
bash peg_grasp_v2.sh checkpoint8-video
```

La riduzione dei vincoli V41 è terminata al livello 50%. Il nuovo lavoro separato riguarda velocità e randomizzazione sullo stesso piolo:

```bash
watch -n 5 cat logs/grasp_v2/autonomy_motion_positions_v42/status.json
```

### Velocità e piccoli spostamenti sullo stesso piolo — V42

Checkpoint8 resta riproducibile alla velocità originale. **V42 è concluso:** con ±0,25 mm / ±1° ottiene **31/32 successi** (16/16 e 15/16). Le velocità maggiori sono state respinte: **13/16** con 1,5× in aria / 1,25× in contatto e **11/16** con 1,5× solo in aria. Si conserva quindi la velocità originale. Pinze, tempi di stabilità, limiti dei giunti e forze ammesse restano invariati.

A ±0,5 mm / ±2° conclude **30/32** (16/16 e 14/16): non supera la soglia di 15/16 per ciascun seed. I due fallimenti riguardano la chiusura di PSM2 durante il transfer; la presa e il lift di PSM1 riescono in tutti i casi. Il curriculum si arresta senza ampliare ulteriormente le perturbazioni e senza riaddestrare l'approccio PSM1. Report: [stato V42](logs/grasp_v2/autonomy_motion_positions_v42/status.json). Sono risultati **assistiti**, con molla di presa e guida del deposito.

Il peg rimane su L5; board e pioli non si spostano. Al reset si decentrano il foro rispetto al piolo e si ruota il peg attorno al foro. La destinazione resta R2. I livelli sono:

| Livello | Decentramento massimo per asse XY | Rotazione |
| --- | --- | --- |
| 0 | 0 mm | 0° |
| 1 | ±0,25 mm | ±1° |
| 2 | ±0,50 mm | ±2° |
| 3 | ±0,75 mm | ±3,5° |
| 4 | ±1,00 mm | ±5° |

Ogni aumento richiede almeno **15/16 task completi per ciascuno dei due seed**, con la stessa policy. L'assistenza resta al 100% durante questo confronto. Se peggiora l'approccio, il programma raccoglie esempi assistiti e prova un fine-tuning BC limitato ai movimenti di approccio PSM1. Presa, pinze e passaggio fra i PSM restano protetti; servono nuove prove fisiche anche sulle condizioni precedenti. Un errore di handover/deposito arresta l'allargamento e viene conservato per una correzione mirata.

Questi comandi usano l'ultima policy verificata, alla velocità originale; `--randomized` usa il livello superato di ±0,25 mm / ±1°:

```bash
conda activate PegTransfer
bash peg_grasp_v2.sh autonomy-motion-play
# Prova le perturbazioni dell'ultimo livello superato:
bash peg_grasp_v2.sh autonomy-motion-play --randomized
# Registra da SSH:
bash peg_grasp_v2.sh autonomy-motion-play --video
```

V43 ha registrato osservazioni e azioni durante il transfer sui casi del seed 9876, senza cambiare pesi o criteri di successo. Per seguirne la diagnosi:

```bash
tail -f logs/grasp_v2/autonomy_handover_diagnostic_v43_startup.log
cat logs/grasp_v2/autonomy_handover_diagnostic_v43/diagnostic_status.json
```

Non serve rilanciare V42. Per un futuro esperimento separato, scegli una cartella di output nuova:

```bash
bash peg_grasp_v2.sh autonomy-motion-curriculum \
  --policy .checkpoint_runs/checkpoint8-assisted-full-bc-constraints/model/policy.pt \
  --output logs/grasp_v2/autonomy_motion_positions_NEW
```

### Correzione della presa PSM2 — V47/V48

V43 conferma **14/16 task completi**, con un timeout e una caduta nella chiusura PSM2. Nel timeout la rete smette di chiudere con forze di circa 0,12/0,18 N: PSM2 è vicino al proprio obiettivo, ma la presa resta debole. Presa e lift PSM1 riescono in tutti i 16 casi.

La prima prova V45 usava una presa PSM2 diversa ed è stata interrotta prima del training V46. V47 usa la stessa geometria del checkpoint: parete300°, profondità4 mm, tilt0°, apertura0,25 rad. V47 raccoglie recuperi: la rete esegue l'approccio fino al passo 2650; poi l'insegnante riparte dal peg realmente sollevato e completa il transfer/deposito. È raccolta supervisionata, non una valutazione autonoma. V48 attende dati completi e chiusura corretta, poi allena una rete aggiuntiva per la pinza PSM2 durante chiusura/rilascio del donatore: **BC, 2.000 aggiornamenti CUDA, lr 0,00005, batch 1.024**. I pesi già esistenti, compreso il pickup PSM1, restano identici. L'insegnante non viene aggiunto all'inference.

Il candidato passa prima il pilot fisso 3/3 e le condizioni precedenti sui seed123/9876, poi ±0,5 mm / ±2° su due seed nuovi, 2468/54321. Servono almeno 15/16 per ciascun seed; vincoli e criteri fisici restano invariati. Nessuna sostituzione automatica di checkpoint8. **67 controlli CUDA superati**; il controllo con due aggiornamenti sintetici verifica solo l'isolamento dei pesi, non il successo fisico.

**V48 è concluso e verificato:** pilot fisso 3/3; ±0,25 mm / ±1°:16/16 e15/16; ±0,5 mm / ±2° sui seed nuovi2468/54321:15/16 e16/16, cioè **31/32 task completi assistiti**. È salvato come **checkpoint9**, con pesi, sorgenti originali, cinque report e ricevute di chiusura verificati. È BC; molla di presa e guida al deposito restano attive durante il task. [Report conservato](milestones/checkpoint9-assisted-full-bc-randomized.json).

Per vedere tre episodi con camera frontale dal desktop della workstation:

```bash
bash peg_grasp_v2.sh checkpoint9-play --episodes 3
```

Per un video da SSH, un episodio:

```bash
bash peg_grasp_v2.sh checkpoint9-video
```

### Spostamenti più rapidi, interazioni lente — V49/V50

Con il controllo precedente, anche trasportare il peg già preso era limitato a3 mm/s. V49 confronta gli stessi pesi V48 con un nuovo profilo opzionale: spostamenti liberi fino a50/75/100 mm/s; trasporto con singola presa stabile fino a9/12 mm/s. Rallenta vicino al peg, alla board o all'altro braccio, e quando la presa è debole o scivola. Presa, handover e inserimento mantengono il limite di contatto3 mm/s; pinze, tempi di stabilità, vincoli e criteri di successo restano invariati. Queste velocità sono limiti del riferimento: il simulatore misura il risultato reale.

**V49 è concluso e scartato:** baseline3/3; ciascuno dei tre profili rapidi0/3, tutti timeout dopo la presa PSM1 senza lift stabile. Nessun profilo selezionato. I report negativi e il codice della prova sono conservati. Il rallentamento senza contatto e le rotazioni introdotti dal profilo `clearance` cambiavano anche l'avvicinamento appreso: queste modifiche non sono confermate dal simulatore.

**V50 è concluso e verificato:** il profilo `travel` a2× conserva31/32 successi sugli stessi reset di V48, con15/16 sul seed2468 e16/16 sul seed54321, più3/3 fissi. Il tempo medio simulato passa da51,97 a47,64 s: **−8,3%**. Mantiene i limiti rotazionali1 rad/s libero/0,2 rad/s in contatto e il minimo originale25 mm/s nell'approccio senza contatto. Velocità massime selezionate: libero50 mm/s e trasporto stabile6 mm/s; contatto3 mm/s. Livelli3× e4× scartati perché perdono casi riusciti. Pinze, vincoli e criteri restano identici. Non c'è stato un nuovo training: stessi pesi BC V48. Tempo simulato e durata reale sulla workstation restano misure diverse.

Ogni candidato deve completare3/3 episodi fissi, almeno15/16 per seed2468/54321 con ±0,5 mm / ±2°, conservare ogni caso riuscito di V48 e ridurre il tempo medio simulato di almeno2%. La velocità scelta è quella con il miglior tempo medio tra le prove accettate. Il comando di replay segnala le prove ancora in corso o scartate senza traceback e non esegue un profilo fallito. **78 controlli CUDA passati; dieci report fisici e ricevute di chiusura verificati.**

### checkpoint10: full task with faster transit and assisted grasp

Il profilo V50 selezionato è conservato con pesi, sorgenti congelate, configurazione2× e report verificati,52 file protetti tramite SHA256. Checkpoint9 conserva V48 con la velocità originale. [Report checkpoint10](milestones/checkpoint10-assisted-full-bc-transit.json).

Per vedere tre episodi con la velocità verificata, dal desktop della workstation:

```bash
bash peg_grasp_v2.sh checkpoint10-play --episodes 3
```

Per registrare un episodio da SSH:

```bash
bash peg_grasp_v2.sh checkpoint10-video
```

```bash
watch -n 5 cat logs/grasp_v2/autonomy_transit_speed_v50/status.json
```

Il comando del confronto V50 seleziona ora lo stesso profilo verificato:

```bash
bash peg_grasp_v2.sh autonomy-transit-play --episodes 3
```

Per un video del profilo verificato:

```bash
bash peg_grasp_v2.sh autonomy-transit-play --episodes 1 --video
```

### SAC v2: apprendere i movimenti senza sequenza programmata

Nuovo esperimento separato: la rete decide movimento XYZ, rotazione e ganascia di **entrambi** i PSM. Non riceve fasi o waypoint; conosce lo stato fisico e la destinazione finale. Restano IK, limiti di movimento e contatto. **Il vincolo sul peg è disattivato**, anche durante il training.

Si parte dalle dimostrazioni del checkpoint3, poi SAC aggiorna actor e critic usando i risultati fisici. I dati con vincolo servono soltanto all'imitazione; il critic iniziale usa il pickup fisico senza vincolo. Il basso errore di imitazione non certifica un episodio riuscito. Il primo tentativo con comandi di velocità (`sac_free_lift_v1`) ha dato 0/16 pickup. Anche il secondo tentativo con comandi di posa (`sac_free_lift_v2`) è terminato: **0/16 pickup, 16 timeout e nessun contatto rilevato con il peg**. Entrambi restano conservati come risultati negativi; il gate per passare all'handover non è superato. [Report del secondo tentativo](logs/grasp_v2/sac_free_lift_v2/evaluation.json).

Il curriculum è **presa/lift → handover → task completo**. Ogni passaggio richiede almeno 16 episodi e ≥90% di successi fisici con lo stesso checkpoint. Questo primo esperimento riguarda L5 → R2 fisso, senza domain randomization; non è ancora un nuovo checkpoint full riuscito.

Parametri iniziali: CUDA con 16 ambienti, rete 256+256, batch 1024, learning rate 0,0001, gamma 0,9995, entropia SAC 0,00001, 12000 aggiornamenti BC e 1000 aggiornamenti iniziali del critic. Replay su GPU; un aggiornamento SAC per passo vettoriale. La loss dell'actor unisce SAC e BC con peso 80, per evitare che SAC cancelli subito il comportamento imitato. Le azioni indicano una posa relativa allo strumento e un'apertura assoluta della pinza: piccoli errori non si accumulano integrando comandi di velocità. La reward premia una sola volta presa, lift stabile e handover; il successo finale richiede foro sul post e deposito libero stabile per 1 s.

```bash
conda activate PegTransfer

# Primo esperimento: 100.000 transizioni, 40 s massimi per episodio.
# Il nome della cartella deve essere nuovo; i risultati esistenti non vengono sovrascritti.
bash peg_grasp_v2.sh sac-train --skill lift --headless \
  --num_envs 16 --steps 100000 --episode_steps 4000 --episodes 16 --bc_steps 12000 \
  --output logs/grasp_v2/sac_free_lift_v2

# Ultime righe del training terminato.
tail -n 40 logs/grasp_v2/sac_free_lift_v2_startup.log
# tail -f resta in attesa di nuove righe anche dopo la fine: uscire con Ctrl+C.

# Inference diagnostica del candidato fallito: final.zip è già disponibile.
bash peg_grasp_v2.sh sac-play \
  --checkpoint logs/grasp_v2/sac_free_lift_v2/final.zip \
  --skill lift --num_envs 1 --episodes 1 --realtime

# Prova dell'intera sequenza: misura ciò che riesce realmente senza vincolo.
bash peg_grasp_v2.sh sac-evaluate \
  --checkpoint logs/grasp_v2/sac_free_lift_v2/final.zip \
  --skill full --num_envs 16 --episodes 16 --headless

# Solo dopo il superamento del gate lift:
bash peg_grasp_v2.sh sac-train --skill handover --headless \
  --checkpoint logs/grasp_v2/sac_free_lift_v2/final.zip \
  --steps 300000 --episodes 16 --output logs/grasp_v2/sac_free_handover_v1

# Solo dopo il superamento del gate handover:
bash peg_grasp_v2.sh sac-train --skill full --headless \
  --checkpoint logs/grasp_v2/sac_free_handover_v1/final.zip \
  --steps 500000 --episodes 16 --output logs/grasp_v2/sac_free_full_v1
```

`sac-play --video` registra la visuale frontale e funziona da SSH; usare `--num_envs 1 --episodes 1`. Ogni modello salva `pretrained.zip`, `latest.zip`, `final.zip`, report e sorgenti; il replay usa i sorgenti del modello. Se la verifica fallisce, si conserva il risultato e si corregge prima di avanzare nel curriculum. Checkpoint4 rimane disponibile con `full-play`.

- [Report breve e timeline](PEG_TRANSFER_TIMELINE.md).
- [Guida tecnica e comandi](PEG_TRANSFER.md).
- [Valutazione del checkpoint completo](logs/sac_sequence/eval_full_unguided_2026-10-02_15-11-38.json).
- [Checkpoint](logs/sac_sequence/2026-10-02_15-11-19_sequence_bc/final.zip).

I risultati SAC sono conservati in `logs/sac/` e `logs/sac_sequence/`: report, configurazioni, dimostrazioni, checkpoint e video. I grandi replay buffer `.pkl`, cache, asset scaricati e la dipendenza locale Isaac Lab restano fuori da Git. I risultati PPO precedenti restano nella workstation in `logs/rsl_rl/`.

## checkpoint1: task success but grasping not optimal

Versione di riferimento: **100/100 trasferimenti L5 → R2**, rete addestrata per imitazione, presa ancora assistita. La pinza e il movimento sono comandati dalla rete; il codice mantiene il peg agganciato dopo la presa. Miglioreremo il grasp nelle prossime versioni.

Modello e codice originale sono conservati nel commit `7a53c10` e nel tag `checkpoint1`. Il comando dedicato usa una copia separata del task e degli asset locali, quindi le modifiche future al progetto e i nuovi training non sostituiscono questa versione. Richiede l'ambiente `PegTransfer` originale (Isaac Sim 5.1, Isaac Lab 0.48.0, SB3 2.9.0). Gli asset scaricati e le dipendenze Isaac vanno conservati sulla workstation; il launcher controlla modello, codice, asset locali e versioni prima di partire. Non garantisce la stessa esecuzione con librerie o asset robot diversi.

Dal repository, con `conda activate PegTransfer`, aprire la GUI **sul desktop della workstation**:

```bash
bash peg_checkpoint1.sh play
```

Registrare un video **anche da SSH**:

```bash
bash peg_checkpoint1.sh video
```

Video e nuovi report sono in `logs/checkpoint1/`; i video nella sottocartella `sequence_videos/`. Ogni esecuzione ha un nome diverso.

Controllare il salvataggio senza avviare Isaac Sim:

```bash
bash peg_checkpoint1.sh check
```

La copia separata è in `.checkpoint_runs/checkpoint1/`: **conservarla**, soprattutto gli asset in `data/`. Se si clona il repository su un'altra macchina, ripristinare prima ambiente e asset originali; il comando ricrea il codice dal commit salvato. I dettagli del modello sono in [milestones/checkpoint1.json](milestones/checkpoint1.json).

## v2: grasp con contatti fisici

La v2 sperimenta prese sui vertici e sulla parete del foro, usando collider ricavati dalla mesh, attrito e contatti elastici. Le prove singole di presa non agganciano artificialmente il peg alla pinza. **Parete PSM1: 1/1 prime prove riuscite**, con lift di circa 35 mm, tenuta di 2 s e rilascio. Salvati [risultato e codice della prova](logs/grasp_v2/wall_success1_2026-10-05.json). Sul vertice l'approccio riesce, ma anche la nuova centratura termina per spostamento del peg durante la chiusura: 0/1. La deformazione della gomma resta da modellare e calibrare.

Sul desktop della workstation, nell'ambiente `PegTransfer`:

```bash
bash peg_grasp_v2.sh tests
bash peg_grasp_v2.sh probe --mode vertex --episodes 1 --realtime
bash peg_grasp_v2.sh probe --mode wall --episodes 1 --realtime
```

[Geometria, criteri di presa e parametri v2](PEG_GRASP_V2.md). Nuovi risultati in `logs/grasp_v2/`. Checkpoint1 continua a usare il proprio codice conservato.

Le prese v2 usano ora ingressi inclinati compatibili con il polso. Prima di avviare Isaac, il comando verifica l'arrivo delle dita e un percorso di 72 punti entro i limiti originali dei giunti. Questo controllo cinematico è superato per le prese predefinite su entrambe le pinze; la nuova presa sul vertice e le prese PSM2 restano da verificare nella simulazione.

La raccolta wall PSM1 ha completato **6/6 prese fisiche**. Il primo modello `wall_bc1` si arresta prima del peg, nella fase `above`, a circa 6,7 mm dal waypoint. È conservato per confronto.

`wall_bc2` arriva al peg, ma resta in chiusura con pressione instabile. `wall_bc3` aggiunge feedback sulla forza: la rete regola la risposta, ma non può smettere di chiudere mentre manca pressione. Camera frontale inclinata, centrata su entrambi i PSM e sulla board.

`wall_bc3` è già stato addestrato sulla **RTX 6000** e salvato in `logs/grasp_v2/wall_bc3/policy.npz`. Anche l'inference neurale usa CUDA. Il comando per riprodurre il training dalle sei dimostrazioni salvate, usando una nuova cartella di output, è:

```bash
bash peg_grasp_v2.sh train --device cuda:0 --demonstrations logs/grasp_v2/wall_demos_bc1.npz --initialize_from logs/grasp_v2/wall_bc2/policy.npz --output logs/grasp_v2/wall_bc3_repeat
```

Inference visiva:

```bash
bash peg_grasp_v2.sh probe --mode wall --policy logs/grasp_v2/wall_bc3/policy.npz --device cuda:0 --episodes 1 --realtime
```

È un controllo ibrido: feedback di posizione/pressione, scelta della presa, orientamento e fasi restano programmati. Geometria, coppia e attrito sono conservati. **Policy verificata in Isaac: 3/3 episodi completi di presa, lift, tenuta, abbassamento e rilascio, senza agganci assistiti.** Il riuso dei dati richiede lo stesso codice fisico; il launcher verifica questa compatibilità. Parametri BC: 4000 aggiornamenti, batch 1024, rete 128+128, learning rate 0,0001. Dati e campionamento restano sulla GPU durante il training. I modelli precedenti sono conservati.

Per regolare l'inquadratura: `--camera_eye X Y Z --camera_target X Y Z`. Default: camera `(0.025, +0.32, 0.25)`, target `(0.025, 0, 0.035)` in metri: sul lato opposto dei PSM, inclinata verso la board. Il comando `probe --policy` usa i sorgenti originali salvati con il modello; le modifiche successive al task completo non alterano questa presa.

Il controllo della presa richiede pressione stabile su entrambe le dita prima del lift, verifica l’attrito realmente usato da PhysX e ferma il movimento se il peg scivola. Le sei dimostrazioni della parete usano 0,35 N per dito, attrito 0,9 e coppia massima 0,015 N·m. Questi valori sono sperimentali e non calibrati su gomma reale.

Basato sul workflow i4h per robotic surgery e sul lavoro locale dVRK. Le licenze originali sono conservate.

## checkpoint2: wall grasp fisico PSM1 riuscito

`wall_bc3`: **3/3 episodi completi**, lift circa 34,7–34,8 mm, tenuta almeno 2 s, slittamento massimo durante la tenuta 0,025 mm. Forza circa 0,35/0,31 N sulle due dita. Nessun aggancio artificiale. Il controllo resta ibrido: la rete regola la risposta, il codice sceglie presa e fasi. Questo checkpoint riguarda la presa PSM1, non ancora il trasferimento completo.

[Salvataggio e risultati](milestones/checkpoint2-wall-bc3.json). Modello e sorgenti originali: `logs/grasp_v2/wall_bc3/`. Conservare anche ambiente e asset Isaac originali.

Rivedere l'inference sul desktop della workstation:

```bash
bash peg_grasp_v2.sh probe --mode wall --policy logs/grasp_v2/wall_bc3/policy.npz --device cuda:0 --episodes 1 --realtime
```

Registrare un nuovo video da SSH, usando la visuale frontale:

```bash
bash peg_grasp_v2.sh video --device cuda:0
```

[Video già registrato: un episodio, visuale frontale](logs/grasp_v2/videos/wall_bc3_front_verified_2026-10-05.mp4), 1280×720, 25 fps, 29,84 s. Rendering RTX e codifica NVIDIA NVENC. [Report della presa nel video](logs/grasp_v2/wall_bc3_front_verified_2026-10-05.json).

Il task fisico completo è un esperimento separato (`bash peg_grasp_v2.sh full`), con comando esplicito di entrambe le pinze: PSM1 mantiene la pressione mentre PSM2 arriva, e apre solo dopo la conferma dei due contatti PSM2. L'obiettivo è il deposito su R2, senza agganci assistiti. I risultati delle prove sono documentati nella timeline.

**Prime prove del full task: 0/6 trasferimenti.** PSM1 completa presa, lift e movimento verso la zona di transfer nelle quattro prove con presa originale. PSM2 arriva, ma durante la chiusura fa scivolare la presa donatrice. Una presa PSM1 più profonda, anche bilanciata, supera il limite di forza durante il lift. Non è ancora disponibile una nuova policy full fisica: prima serve un handover stabile. [Report delle sei prove e verifica del video](logs/grasp_v2/front_video_full_task_audit_2026-10-05.json).

## checkpoint3: full task con vincolo temporaneo sul peg

**3/3 trasferimenti L5 → R2 riusciti sulla RTX 6000**, più un episodio riuscito nel video frontale. PSM1 approccia, prende e solleva il peg; PSM2 lo riceve, lo trasporta e lo deposita sul nuovo post. Errore finale del foro: **0,30–0,43 mm**.

La semplificazione è un giunto fisso PhysX fra peg e pinza durante transfer, handover e inserimento. Viene tolto prima del rilascio su R2: il peg deve restare libero e stabile sulla board per 1 s, con entrambe le pinze staccate. Il lift iniziale PSM1 resta fisico e senza sostegno. PSM2 deve avere due contatti con pressione stabile prima che PSM1 apra.

È il riferimento riuscito **prima del training completo**: rete `wall_bc3` per la presa PSM1, controller programmato per la sequenza restante. La presa PSM2 senza vincolo scivola ancora. Scelta autonoma fra prese e randomizzazione delle posizioni restano da sviluppare.

[Report dei tre episodi](logs/grasp_v2/full_assisted_validation3_2026-10-05.json), [verifica delle tracce](logs/grasp_v2/full_assisted_validation3_audit_2026-10-05.json), [salvataggio della versione](milestones/checkpoint3-full-assisted.json), [video frontale completo](logs/grasp_v2/videos/full_assisted_front_2026-10-05.mp4).

Rivederlo nella finestra Isaac sul desktop della workstation:

```bash
bash peg_grasp_v2.sh full-replay --device cuda:0 --episodes 1 --realtime
```

Registrare un nuovo video anche da SSH:

```bash
bash peg_grasp_v2.sh full-replay --device cuda:0 --video
```

Controllare il salvataggio:

```bash
bash peg_grasp_v2.sh full-replay --check
```

Il replay usa la copia conservata in `.checkpoint_runs/checkpoint3-full-assisted/`: codice della prova, scena, modello donatore e asset locali. **Conservare questa cartella e l'ambiente Isaac originale.** Modificare i sorgenti correnti non cambia il replay. Nuovi video e report ricevono nomi diversi in `logs/grasp_v2/`. Questo riferimento e i checkpoint precedenti restano distinti.

## Training della rete sul task completo v2

**Checkpoint4 `full_bc2`: 3/3 trasferimenti riusciti in Isaac**, errore finale del foro **0,40–0,42 mm**. Le tracce confermano presa PSM1 e lift iniziale senza sostegno, due contatti PSM2 stabili prima dell'handover, rimozione del vincolo prima del rilascio e 1 s di deposito libero. Forza massima misurata: **0,65 N**, sotto il limite di 1 N. Questo risultato riguarda la scena fissa L5 → R2.

[Report](logs/grasp_v2/full_bc2_validation3_2026-10-05.json), [audit](logs/grasp_v2/full_bc2_validation3_audit_2026-10-05.json), [salvataggio checkpoint4](milestones/checkpoint4-full-bc2-assisted.json).

Usiamo **Behavior Cloning**: la rete impara dagli episodi completi riusciti. Conserviamo la rete PSM1 `wall_bc3`, perché riapprenderne la presa aveva provocato uno scivolamento nel primo tentativo. Una seconda rete apprende i guadagni del movimento XYZ e delle ganasce per transfer, PSM2 e deposito. Entrambe girano sulla RTX 6000.

È un **controllo ibrido**: ordine delle fasi, obiettivi, orientamento/IK e controlli di contatto restano programmati. Il vincolo temporaneo sul peg resta attivo durante il trasferimento e viene tolto prima del rilascio. Scelta della presa, tenuta PSM2 senza vincolo e posizioni casuali sono passaggi successivi.

Da un terminale nell'ambiente `PegTransfer`, esportare gli esempi già verificati e allenare un nuovo modello:

```bash
# Usare nomi nuovi: i comandi proteggono i file già salvati.
bash peg_grasp_v2.sh full-export --output logs/grasp_v2/full_demos_next.npz
bash peg_grasp_v2.sh full-train \
  --demonstrations logs/grasp_v2/full_demos_next.npz \
  --output logs/grasp_v2/full_bc_next \
  --device cuda:0 --updates 6000 --batch_size 1024 --learning_rate 0.0001
```

Sono **17.930 campioni da tre episodi**, con tutte le 20 fasi. Due episodi sono usati per il training; il terzo è riservato alla verifica degli errori. La nuova rete ha due strati da 256 neuroni e campiona le fasi in modo bilanciato. La loss non basta: verificare il modello in Isaac.

```bash
bash peg_grasp_v2.sh full-play \
  --policy logs/grasp_v2/full_bc_next/policy.npz \
  --device cuda:0 --headless --episodes 3
```

Vedere il modello `full_bc2` già allenato nella finestra Isaac sul desktop della workstation:

```bash
bash peg_grasp_v2.sh full-play \
  --policy logs/grasp_v2/full_bc2/policy.npz \
  --device cuda:0 --episodes 1 --realtime
```

Da SSH, registrare invece un video con la camera frontale:

```bash
bash peg_grasp_v2.sh full-play \
  --policy logs/grasp_v2/full_bc2/policy.npz \
  --device cuda:0 --video
```

`full-play` verifica pesi e codice, usa i sorgenti conservati nel modello e la scena congelata del checkpoint3. Conservare la cartella del modello, `.checkpoint_runs/checkpoint3-full-assisted/`, il manifest del checkpoint3 e l'ambiente Isaac originale. I report nuovi vengono salvati in `logs/grasp_v2/`.

Una copia separata del modello riuscito è conservata in `.checkpoint_runs/checkpoint4-full-bc2-assisted/model/`. Per rivedere quella versione anche dopo altri training:

```bash
bash peg_grasp_v2.sh full-play \
  --policy .checkpoint_runs/checkpoint4-full-bc2-assisted/model/policy.npz \
  --device cuda:0 --episodes 1 --realtime
```


### Checkpoint6: sequenza completa con contatti nativi — insegnante

**3/3 sequenze complete senza aggancio al peg**, L5 → R2: PSM1 prende e solleva, PSM2 prende e sostiene da solo per 2 s, poi trasporta e rilascia sulla board. PSM1 resta fermo durante la presa di PSM2; PSM2 entra verticalmente 4 mm sotto il bordo. Il peg si assesta liberamente sul post dopo il rilascio.

Questo risultato usa il controllore delle dimostrazioni: **non è ancora l'inference della rete**. Modello del pickup checkpoint5 e full assistito checkpoint4 restano conservati. [Report nativo](logs/grasp_v2/autonomy_full_wall_fixed_donor_v13/evaluation.json), [manifest conservato](milestones/checkpoint6-native-full-teacher.json).

Per rivedere il controllore dai sorgenti conservati:

```bash
conda activate PegTransfer
python tools/peg_grasp_v2/autonomy_teacher_replay.py --output logs/grasp_v2/native_full_replay_$(date +%Y%m%d_%H%M%S) --realtime
# Da SSH, sostituire --realtime con --headless --video
```

La raccolta ampliata e il primo training BC full sono prove separate. Seguire la verifica della rete:

```bash
cat logs/grasp_v2/autonomy_resume_status.json
```

Il replay singolo dell’insegnante ha evidenziato un placement fuori tolleranza: checkpoint6 documenta i tre episodi originali riusciti, senza garantire tutte le configurazioni. La candidata neurale full e la correzione del placement sono descritte in [PEG_AUTONOMY.md](PEG_AUTONOMY.md).

## Verso checkpoint11: più posizioni e rotazioni

V51 è avviato dalla rete di **checkpoint10**, che rimane conservata. Prima proviamo rotazioni ±5°/10°/20°/30° sullo stesso cilindro; poi offset ±0,75/1/1,1 mm per asse. Infine aggiungiamo le coppie di cilindri: picking a destra e deposito a sinistra nella camera frontale (nomi L e R rispettivamente nel codice). Le nuove condizioni sono ancora da verificare.

Ogni livello richiede almeno15/16 task completi su ciascuno dei due seed, per ogni coppia di cilindri. Al primo livello insufficiente il workflow si ferma: usiamo i casi falliti per BC/DAgger mirato e verifichiamo di nuovo anche le condizioni precedenti. Questa prima esecuzione valuta la rete congelata; non sta ancora allenando. Velocità2× e aiuti alla presa/deposito restano costanti.

```bash
watch -n 5 cat logs/grasp_v2/autonomy_domain_v51/status.json
# Il riferimento già verificato resta disponibile:
bash peg_grasp_v2.sh checkpoint10-play --episodes 3
```

[Strategia e dettagli](PEG_AUTONOMY.md). Per una nuova esecuzione separata: `bash peg_grasp_v2.sh autonomy-domain-curriculum --output logs/grasp_v2/domain_nuovo`. Non avviare una seconda simulazione mentre V51 è in corso.

V51 è ora concluso: **30/32 sul livello iniziale**, con un precedente successo perso sul secondo seed. Le rotazioni maggiori e i nuovi cilindri non sono stati provati. V52 sta ricontrollando il caso con checkpoint10 originale, prima di decidere il training correttivo. Stato: `watch -n 5 cat logs/grasp_v2/autonomy_baseline_audit_v52/status.json`.

V52 è concluso: **16/16 con checkpoint10 originale** sul seed54321. Ora V53 prova soltanto **rotazione±5°**, mantenendo codice archiviato, offset±0,5 mm, velocità e vincoli. Non è nuovo training; la causa della differenza con V51 resta aperta. Avanzamento: `watch -n 5 cat logs/grasp_v2/autonomy_archived_yaw_v53/status.json`.

## Checkpoint11: rotazioni fino a±5°

**31/32 task completi assistiti**, con offset±0,5 mm attorno allo stesso cilindro L5 e deposito su R2. Pesi BC identici a checkpoint10; non è nuovo training. Un episodio fallisce per forza oltre1 N durante il passaggio a PSM2. Pesi, codice e report sono conservati in40 file verificati.

```bash
bash peg_grasp_v2.sh checkpoint11-play --episodes 3
bash peg_grasp_v2.sh checkpoint11-video
```

[Checkpoint11 conservato](milestones/checkpoint11-assisted-full-bc-yaw5.json). Checkpoint10 resta disponibile.

Video frontale di checkpoint11: [checkpoint11.mp4](logs/grasp_v2/videos/checkpoint11.mp4), un episodio completo riuscito con gli aiuti alla presa e al deposito attivi. Durata 47,52 secondi, 1280×720 a 25 fps, codifica NVIDIA NVENC. [Metadati e SHA256](logs/grasp_v2/videos/checkpoint11.json) e [report dell'episodio](logs/grasp_v2/videos/checkpoint11.evaluation.json). Ogni nuovo checkpoint sarà accompagnato da un video con il suo nome.

Dal terminale del Mac, fuori dalla sessione SSH:

```bash
scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/checkpoint11.mp4 ~/Downloads/
open ~/Downloads/checkpoint11.mp4
```

V54 si è concluso alla verifica iniziale con 30/32: un successo precedente perso sul secondo seed ha fermato il curriculum. Rotazioni e offset maggiori non sono stati provati; il checkpoint11 conservato resta invariato. [Report V54](logs/grasp_v2/autonomy_isolated_domain_v54/completion_summary.json).

V55 si è interrotto dopo16/32 episodi:12 task riusciti salvati, errore nel passaggio all'insegnante prima del lift stabile. La raccolta non è dichiarata completa. V56 ha affinato la pinza PSM2, ma conclude **0/3** nel pilot e viene scartato. Checkpoint12 deriva invece dalla successiva strategia su tre cilindri, descritta sotto. [Strategia](PEG_AUTONOMY.md).

```bash
watch -n 5 cat logs/grasp_v2/autonomy_handover_refine_v56/status.json
```

## Checkpoint12: tre picking e tre placement, con presa assistita

La rete BC trasferisce in ordine **L5→R2, L6→R1, L4→R3**. Conserva i controlli di movimento e pinze di checkpoint11, affina la scelta del lift e aggiunge vincoli temporanei alla presa; non è SAC. Nessun insegnante in inference. Risultati e aiuti dichiarati nel [report checkpoint12](milestones/checkpoint12-three-posts-assisted.json).

Verificati **3/3** nel pilot, **12/12** con offset ±0,1 mm e rotazione ±0,5°, e **3/3 consecutivi nel video frontale**. La verifica riguarda queste tre coppie di cilindri. Molla di presa, mantenimento verticale, limitazione del carico sulle pinze e guida di centraggio restano attivi. Pesi, sorgenti e video sono conservati con SHA256; checkpoint10 e checkpoint11 restano invariati.

```bash
# Sulla workstation, nel terminale del desktop:
conda activate PegTransfer
bash peg_grasp_v2.sh checkpoint12-play
# Dal Mac, fuori da SSH, scarica tutti e tre gli episodi:
scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/checkpoint12_three_episodes.mp4 ~/Downloads/
open ~/Downloads/checkpoint12_three_episodes.mp4
```

Il primo episodio completo è conservato anche in `logs/grasp_v2/videos/checkpoint12.mp4`.

## Checkpoint13: spostamenti più rapidi, interazioni lente

Usa gli stessi pesi BC di checkpoint12. Abbiamo accelerato i movimenti liberi e il trasporto con presa stabile; il controllo rallenta quando il peg o l'altro PSM sono vicini. Presa, scambio e inserimento finale mantengono le velocità precedenti. L'apertura scelta dalla rete viene accelerata soltanto per pinze vuote lontane dal peg. Nessun nuovo training o insegnante durante l'inference; gli aiuti temporanei alla presa e al deposito rimangono attivi.

| Limite della traiettoria | Checkpoint12 | Checkpoint13 |
| --- | --- | --- |
| Spostamento libero | 50 mm/s | 100 mm/s |
| Trasporto con presa stabile e spazio libero | 6 mm/s | 24 mm/s |
| Movimento durante il contatto | 3 mm/s | 3 mm/s |

Sono limiti del riferimento, non velocità garantite di ogni giunto. I limiti fisici e i guadagni dei giunti restano invariati. Le prove a velocità superiori hanno causato blocchi o perdita della presa e sono state scartate.

Verificati **3/3** nel pilot, **12/12** nella conferma sugli stessi reset di checkpoint12 (±0,1 mm e ±0,5°) e **3/3** nel video frontale. Media della conferma: **46,30 s contro 53,05 s**, riduzione **12,7%** del tempo simulato. Nel video: **40,42 s / 46,76 s / 56,53 s**. La verifica riguarda queste tre coppie di cilindri e questi reset. [Report e archivio con SHA256](milestones/checkpoint13-three-posts-fast-assisted.json). Checkpoint12 e il suo video restano conservati.

```bash
# Sulla workstation, nel terminale del desktop, dalla cartella del progetto:
conda activate PegTransfer
bash peg_grasp_v2.sh checkpoint13-play

# Dal Mac, fuori da SSH, scarica il video con i tre episodi:
scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/checkpoint13_three_episodes.mp4 ~/Downloads/
open ~/Downloads/checkpoint13_three_episodes.mp4
```

Il primo episodio completo è anche in `logs/grasp_v2/videos/checkpoint13.mp4`. Per registrare un nuovo replay dal modello archiviato, senza sovrascrivere i video conservati: `bash peg_grasp_v2.sh checkpoint13-video`.

Versione leggera dei tre episodi: `logs/grasp_v2/videos/checkpoint13_three_episodes_under5MB.mp4`, **4,67 MB**, HEVC, stessa durata,1280×720 e25fps. Qualità visiva molto simile, ma compressione non lossless; originale conservato. Confrontati tutti i3592 fotogrammi (SSIM0,9958) e controllati dettagli di peg e pinze. Il fileJSON omonimo collega copia, originale, pesi e report tramiteSHA256.

```bash
# Dal Mac, fuori da SSH:
scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/checkpoint13_three_episodes_under5MB.mp4 ~/Downloads/
open ~/Downloads/checkpoint13_three_episodes_under5MB.mp4
```

Per l'upload in Notion prova la copia **MP4/H.264 a8bit**: `logs/grasp_v2/videos/checkpoint13_three_episodes_notion_under5MB.mp4`, **4,90 MB**. Mantiene1280×720,25fps e tutti i3592 fotogrammi; usa un codec diverso dalla copia HEVC precedente e l'indiceMP4 all'inizio del file. Decodifica completa verificata; l'upload effettivo in Notion resta da provare. Compressione non lossless, con possibili artefatti nei movimenti. Originali e metadati SHA256 conservati.

```bash
# Dal Mac, fuori da SSH:
scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/checkpoint13_three_episodes_notion_under5MB.mp4 ~/Downloads/
open ~/Downloads/checkpoint13_three_episodes_notion_under5MB.mp4
```


### Esito BC+SAC V4

Training concluso (12.000 transizioni): baseline12/12, candidato11/12 con un timeout. Nessuna promozione: checkpoint13 resta il riferimento conservato. Video frontale del candidato: `logs/grasp_v2/videos/bc_sac_residual_v4.mp4`, episodio completo riuscito di41,24s; report e metadati SHA256 nei file omonimi `.evaluation.json` e `.json`.

```bash
# Dal Mac, fuori dalla sessione SSH:
scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/bc_sac_residual_v4.mp4 ~/Downloads/
open ~/Downloads/bc_sac_residual_v4.mp4
```


### Diagnosi e rilancio BC+SAC V5

Workflow in corso: `logs/grasp_v2/bc_sac_recovery_retry_v5/status.json`. Prima controlla il recupero anticipato della presa su12 reset, poi registra il video della variante. Se supera il gate, riparte il training in `logs/grasp_v2/bc_sac_residual_v5/`, con confronti fisici, video e seed indipendente. Checkpoint13 resta conservato. I filmati saranno disponibili dopo le rispettive registrazioni, con report e metadati SHA256 omonimi.

```bash
# Dal Mac, fuori da SSH; dopo che lo stato conferma la registrazione:
scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/bc_sac_recovery_retry_v5.mp4 ~/Downloads/
# Video del nuovo candidato, se il training supera i gate iniziali:
scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/bc_sac_residual_v5.mp4 ~/Downloads/
```


### Esito BC+SAC V5

Workflow concluso: precedente timeout corretto, conferma12/12; seed indipendente11/12 con un timeout prima del lift stabile. Non promosso; checkpoint13 resta il riferimento. Video frontale candidato1/1 completo,42s: `logs/grasp_v2/videos/bc_sac_residual_v5.mp4`, report e SHA256 nei file omonimi.

```bash
# Dal Mac, fuori da SSH:
scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/bc_sac_residual_v5.mp4 ~/Downloads/
open ~/Downloads/bc_sac_residual_v5.mp4
```


### Confronto vanilla e safety-tuned

Le policy vanilla e le varianti safety-tuned rimangono sempre separate e conservate, con pesi, configurazioni, report e video distinti. Il confronto usa gli stessi scenari e criteri di safety. Gli aiuti fisici già presenti sono dichiarati anche nella baseline vanilla; eventuali nuovi vincoli sono riportati come parte della variante. La promozione di una variante non sostituisce né cancella la baseline. Protocollo: [studio di safety](docs/SAFETY_LITERATURE_AND_PLAN.md).
