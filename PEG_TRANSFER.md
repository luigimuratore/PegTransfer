# Peg transfer con due PSM — guida del progetto

> File Markdown pronto da importare in Notion. Repository di lavoro: `/home/francesco/Documents/GIGI/i4h-workflows`.

## 1. Cosa fa la task

`Isaac-Peg-Transfer-Dual-PSM-v0` crea due PSM, una board con post e un peg. A ogni episodio il peg parte da **L5 o L6**; il target è **R2**. Una policy PPO controlla i due bracci con IK relativo e le due pinze: **14 azioni** (6+1 per PSM). L'episodio dura al massimo **10 s**.

La scena è indipendente dalle vecchie task `handover`, `lift` e `reach`. I loro file e checkpoint sono conservati in `../archivio_i4h_2026-09-29/`. I vecchi checkpoint hanno osservazioni/azioni diverse e **non vanno caricati** in questa task.

## 2. Dove cambiare il comportamento

I percorsi seguenti partono da `workflows/robotic_surgery/scripts/simulation/`:

| File | Responsabilità |
| --- | --- |
| `exts/robotic.surgery.tasks/robotic/surgery/tasks/surgical/peg_transfer/scene_cfg.py` | Board, post, due PSM nella scena e reset del peg su L5/L6. |
| `exts/robotic.surgery.tasks/robotic/surgery/tasks/surgical/peg_transfer/env_cfg.py` | Azioni, osservazioni, pesi dei reward, terminazioni e durata. |
| `exts/robotic.surgery.tasks/robotic/surgery/tasks/surgical/peg_transfer/mdp.py` | Formule di osservazioni, reward, target R2 e successo. |
| `exts/robotic.surgery.tasks/robotic/surgery/tasks/surgical/peg_transfer/agents/rsl_rl_cfg.py` | Parametri PPO, checkpoint e numero predefinito di iterazioni. |
| `scripts/environments/preview_peg_transfer.py` | Apre la scena senza policy. |
| `scripts/reinforcement_learning/rsl_rl/train.py` / `play.py` | Training / inference. |

## 3. Reward e successo

| Fase | Reward (peso) |
| --- | --- |
| PSM1 raggiunge e trattiene il peg | `psm1_reach` (2), `psm1_hold` (2) |
| Peg sollevato | `lift` (6) |
| PSM2 raggiunge e trattiene il peg sollevato | `psm2_reach` (3), `psm2_hold` (5) |
| Peg verso R2 e posizionamento | `transport` (8), `placement` (5) |
| Peg stabile e rilasciato sul target | `success` (100) |
| Movimenti bruschi | `action_rate` (−0,001) |

Il successo richiede peg vicino a R2, velocità bassa, pinza PSM2 aperta e punte dei due PSM lontane dal peg. **La condizione non dimostra che PSM2 abbia davvero effettuato il passaggio:** il reward lo incentiva, ma la manovra va verificata visivamente.

Per cambiare l'importanza di un comportamento, modifica il suo `weight` in `env_cfg.py`. Per cambiare la misura, modifica la funzione in `mdp.py`. Dopo ogni modifica: prova breve, nuovo `--run_name`, confronto dei reward e del successo in simulazione. Se cambi osservazioni o azioni, addestra una policy nuova.

## 4. Preparare il terminale

Requisiti del progetto: Python 3.11, Isaac Sim 5.1.0, Isaac Lab `release/2.3.0` e RSL-RL. Su questa macchina l'ambiente Conda si chiama `PegTransfer`. L'installazione iniziale è descritta nello script `tools/env_setup_robot_surgery.sh`: eseguirlo **una sola volta** in un ambiente Conda nuovo; non rilanciarlo se `third_party/` esiste senza prima verificare lo stato dell'installazione.

Per ogni nuova sessione:

```bash
cd /home/francesco/Documents/GIGI/i4h-workflows
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate PegTransfer
export PYTHONPATH="$PWD/workflows/robotic_surgery/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=Y
export ACCEPT_EULA=Y
```

## 5. Scena, training e inference

**Vedere la scena senza checkpoint:**

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/environments/preview_peg_transfer.py --num_envs 1
```

**Prova breve della task RL:**

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Peg-Transfer-Dual-PSM-v0 --headless --num_envs 4 --max_iterations 1
```

**Training:**

```bash
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Peg-Transfer-Dual-PSM-v0 --headless \
  --num_envs 64 --max_iterations 3000 --run_name peg_transfer_v1
```

I checkpoint vengono salvati in `logs/rsl_rl/peg_transfer_dual_psm/<run>/`. Per vedere i progressi: `python -m tensorboard.main --logdir logs/rsl_rl/peg_transfer_dual_psm`.

**Inference:** trova un checkpoint, poi sostituisci `NOME_RUN` e `model_XXXX.pt` con i nomi effettivi.

```bash
find logs/rsl_rl/peg_transfer_dual_psm -name 'model_*.pt'
python workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Peg-Transfer-Dual-PSM-Play-v0 --num_envs 1 \
  --load_run NOME_RUN --checkpoint model_XXXX.pt
```

## 6. Stato della verifica

La struttura, la sintassi Python e i percorsi degli asset sono stati controllati. **L'avvio in Isaac Sim e la fisica del peg transfer richiedono ancora una prova reale.** Prima di un training lungo, controlla che presa, target R2, collisioni e condizione di successo corrispondano alla scena visualizzata.
