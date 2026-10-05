# Dual-PSM peg transfer

Progetto Isaac Sim / Isaac Lab per il peg transfer con due dVRK Patient Side Manipulators.

La policy attuale completa il trasferimento **L5 → R2**: 100/100 successi nella configurazione fissa con presa assistita. È stata addestrata per imitazione (Behavior Cloning); la pipeline SAC è disponibile per gli aggiornamenti tramite reward. Sequenziamento e controllo del braccio inattivo restano programmati.

- [Report breve e timeline](PEG_TRANSFER_TIMELINE.md).
- [Guida tecnica e comandi](PEG_TRANSFER.md).
- [Valutazione del checkpoint completo](logs/sac_sequence/eval_full_unguided_2026-10-02_15-11-38.json).
- [Checkpoint](logs/sac_sequence/2026-10-02_15-11-19_sequence_bc/final.zip).

I risultati SAC sono conservati in `logs/sac/` e `logs/sac_sequence/`: report, configurazioni, dimostrazioni, checkpoint e video. I grandi replay buffer `.pkl`, cache, asset scaricati e la dipendenza locale Isaac Lab restano fuori da Git. I risultati PPO precedenti restano nella workstation in `logs/rsl_rl/`.

Basato sul workflow i4h per robotic surgery e sul lavoro locale dVRK. Le licenze originali sono conservate.
