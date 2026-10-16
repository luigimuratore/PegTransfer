# PegTransfer — contesto per riprendere su un altro computer

Aggiornato il16 ottobre2026. Questo file trasferisce il contesto del progetto; non è la cronologia completa della chat.

## Progetto e ambiente

- Repository indicata dall'utente: `https://github.com/luigimuratore/PegTransfer.git`. Non presumere che gli ultimi file locali siano già stati pubblicati su GitHub.
- Workspace Isaac: `/home/francesco/Documents/GIGI/i4h-workflows` sulla workstation `francesco@100.77.29.104` (`mitic-13k`).
- Conda `PegTransfer`, RTX6000 Ada; IsaacSim5.1, IsaacLab0.48, PyTorch2.7 CUDA.
- Task: PSM1 prende il peg triangolare forato, solleva e trasferisce aPSM2; PSM2 deposita su un altro cilindro.

## Riferimento da preservare

**Checkpoint13** è la baseline congelata. BC strutturata con IK di basso livello, osservazioni fisiche privilegiate e aiuti temporanei di presa/deposito; non è una policy visiva né una validazione di autonomia fisica senza aiuti.

- Manifest: `milestones/checkpoint13-three-posts-fast-assisted.json`.
- Pesi: `.checkpoint_runs/checkpoint13-three-posts-fast-assisted/model/policy.pt`.
- SHA256 pesi: `956e60f3a3e8ae424289a247cc448105be749f6134633ffc515db02d5b8059c6`.
- Video: `logs/grasp_v2/videos/checkpoint13_three_episodes.mp4`.
- Comando nella repository: `bash peg_grasp_v2.sh checkpoint13-play`.
- Ultima verifica:55 file archiviati integri. Non modificare l'archivio per adattarlo alla nuova scena.

Esperimento BC+SAC V5 concluso:12/12 sul confronto appaiato,11/12 su seed indipendente; un timeout prima del lift stabile. Non promosso. Nessun nuovo training avviato dopo il cambio di strategia. Controllare comunque processi e stato attuali prima di rilanciare qualcosa.

## Nuova strategia concordata

1. Registrare dimostrazioni umane sul dVRK reale.
2. Convertire i dati ROS in dataset sincronizzato immagini/azioni, calibrato rispetto alla scena Isaac.
3. Allenare una **nuova policy IL visiva vanilla**.
4. Imparare e verificare una reward dalle dimostrazioni/preferenze; BC di per sé non richiede una reward.
5. Conservare la vanilla e creare una **variante safety-tuned separata**.
6. Confrontare vanilla e safety-tuned con gli stessi scenari/metriche; mantenere checkpoint13 come riferimento storico, dichiarando le diverse osservazioni e gli aiuti.

**Safety prioritaria rispetto al success rate**: un task completato violando i criteri di safety è non accettabile. Conservare tutti gli esiti, anche fallimenti e timeout. Una reward favorevole non sostituisce una valutazione indipendente.

## Setup reale confermato dall'utente

- ROS1, dVRK precedente; distribuzione/versione esatta non ancora identificate.
- Endoscopio **monoculare** che pubblica su un topic ROS.
- Nessun sensore di forza dedicato alle pinze. Segnali motori/effort forse disponibili, da verificare; non considerarli forze di contatto validate.
- Scena fisica già disponibile.

**Prossima azione:** inventario dei topic sul computer dVRK, poi un solo bag pilota completo prima di molte dimostrazioni.

```bash
rosversion -d
rosversion dvrk_robot
rostopic list -v
```

Registrare immagini con timestamp, pose/giunti misurati e setpoint di entrambi iPSM, aperture pinze, calibrazioni/trasformazioni e stato/eventi utili. Verificare nomi e tipi reali, comprese API legacy. Conversione offline: non serve collegare direttamente ROS1 al simulatore per il training. Non ripubblicare un bag di comandi sul robot reale.

## Ultima modifica alla scena Isaac corrente

Tutti i12 cilindri sono ora alti **12mm dalla superficie della board alla punta**. Board topZ10mm, centro cilindriZ16mm, puntaZ22mm; raggio2mm e coordinateXY invariati. Aggiornate anche le quote dei controlli geometrici SAC. Verifica geometrica USD in Isaac/CUDA riuscita sui12 cilindri. I checkpoint congelati conservano la geometria originale.

Audit: `logs/scene_audits/posts_12mm_20261016_102947_526048.json`.

## Documenti da leggere

- `GUIDA_DVRK_REALE.txt`: guida breve per il computer del robot.
- `docs/DVRK_REAL_TO_ISAAC_PLAN.md`: registrazione, sincronizzazione, calibrazione, conversione, replay e training.
- `docs/SAFETY_LITERATURE_AND_PLAN.md`: letteratura, metriche e protocollo vanilla/safety.
- `AGENTS.md`: ogni checkpoint conservato deve avere un video completo frontale dai propri pesi/sorgenti/configurazione, report e metadatiSHA256; non sovrascrivere gli originali.

## Testo per la nuova chat

> Leggi CONTESTO_PEGTRANSFER.md e GUIDA_DVRK_REALE.txt. Riprendiamo dal primo bag pilota sul dVRK ROS1 reale con endoscopio monoculare. Checkpoint13 resta congelato, le policy vanilla e safety-tuned restano separate e la safety è prioritaria. Verifica prima gli output ROS e i dati disponibili; non avviare training o comandi di movimento basandoti su topic o calibrazioni presunti.
