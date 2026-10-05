# Dual-PSM peg transfer

Progetto Isaac Sim / Isaac Lab per il peg transfer con due dVRK Patient Side Manipulators.

La policy attuale completa il trasferimento **L5 → R2**: 100/100 successi nella configurazione fissa con presa assistita. È stata addestrata per imitazione (Behavior Cloning); la pipeline SAC è disponibile per gli aggiornamenti tramite reward. Sequenziamento e controllo del braccio inattivo restano programmati.

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

Basato sul workflow i4h per robotic surgery e sul lavoro locale dVRK. Le licenze originali sono conservate.
