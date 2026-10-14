# PegTransfer: letteratura e piano di valutazione della safety

Data: 14 ottobre 2026. Questo documento propone uno studio; non introduce nuovi criteri di successo, training o modifiche ai checkpoint.

## Requisiti aggiornati dall'utente

**Policy vanilla e safety-tuned sempre separate.** Conservare il modello di riferimento senza modificarne pesi o configurazione quando si prepara una variante safety. Ogni variante ha cartella, identificativo, pesi, sorgenti/configurazione, report e video propri, collegati tramite SHA256. Non sovrascrivere o sostituire la vanilla con una variante ottimizzata.

Distinguere due assi: algoritmo (`BC`, `BC+SAC`, poi versioni visive) e trattamento (`vanilla`, `safety_tuned`). La vanilla non è necessariamente priva degli aiuti fisici già presenti: elencare esplicitamente molla di presa, guida al deposito e protezioni del controllo in entrambe. I checkpoint esistenti restano baseline assistite; non rinominarli come policy prive di vincoli.

Per isolare l'effetto del tuning confrontare vanilla e safety-tuned dello stesso algoritmo, con identiche osservazioni/azioni, reset, seed di valutazione, fisica e aiuti di base. Dichiarare esattamente cosa cambia: reward/costi di safety, iperparametri, dati o protezioni aggiuntive in inference. Se cambia anche un vincolo fisico o un filtro del controllo, riportare il confronto come effetto combinato e, quando possibile, aggiungere un'ablation che isoli il vincolo. Registrare seed e budget di training; se differenti, non attribuire tutte le differenze alle sole metriche di safety.

Il valutatore esterno applica gli stessi criteri di safety a entrambe, indipendentemente dalla reward usata nel training. Il benchmark di test rimane separato dal tuning. Includere tutti gli episodi, fallimenti e timeout; conservare anche le varianti che peggiorano. Una promozione cambia il modello raccomandato, non cancella la baseline o gli esperimenti precedenti.

**Safety come criterio principale.** Il completamento è secondario: un episodio completato con una violazione dei criteri di safety è non accettabile. Definire i criteri e valutarli prima di confrontare successo, velocità o reward. Non compensare una violazione con un buon tempo o una riuscita. Riportare frequenza, gravità e durata degli eventi su tutti gli episodi, anche incompleti. Fra configurazioni che superano i criteri, confrontare robustezza e tempo; una policy immobile non è una soluzione, quindi mantenere anche un requisito separato di avanzamento/completamento.

**Policy finale image-based per il sim-to-real.** L'attuale policy a stato fisico è un riferimento di simulazione. La futura policy deve ricavare le decisioni dalle immagini, con storia temporale e una destinazione specificata senza coordinate perfette del simulatore. Pose esatte del peg, contatti PhysX, flag di progresso e stato delle molle non devono diventare input privilegiati della policy da trasferire. La cinematica di basso livello e l'eventuale uso di sensori reali vanno esplicitati separatamente nell'architettura, senza presumerne la disponibilità come osservazioni della policy.

In training il simulatore può fornire annotazioni, dimostrazioni dell'esperto e, se adottato, un critic privilegiato; in inference tali informazioni non entrano nell'actor visivo. Il valutatore di safety rimane separato: in simulazione usa la fisica come riferimento. Sul robot le forze effettive richiedono misure reali o stime validate; la sola immagine non garantisce il rispetto di una soglia di forza. I limiti basati su forza, slip e collisione vanno quindi distinti in misurati, stimati e non osservabili.

Percorso previsto: misure di safety verificabili sull'attuale baseline → dataset di immagini sincronizzato con azioni ed eventi → BC visiva con memoria temporale → confronto con il riferimento a stato → eventuale SAC visivo dopo i gate di safety. Includere randomizzazione di camera, luce, materiali e occlusioni, oltre alle pose; testare separatamente situazioni non viste. Nessuna semplice sostituzione del vettore di stato con una singola immagine è assunta sufficiente.

## Domanda dello studio

Il robot completa il trasferimento con interazioni controllate, una presa stabile e pochi interventi di recupero? BC+SAC migliora questi aspetti rispetto a BC, a parità di fisica e condizioni iniziali?

Separiamo tre risultati: **riuscita del task**, **qualità del movimento**, **eventi di rischio nel simulatore**. Un task veloce o riuscito può includere urti, perdita di pressione o un recupero assistito. Un timeout indica mancata conclusione: non dimostra automaticamente un danno.

## Cosa suggeriscono i lavori forniti

I file di Ross e Rajeswaran sono duplicati; i due Brown sono copie dello stesso lavoro. Consideriamo sette lavori e il dataset JIGSAWS.

| Lavoro | Idea principale | Utilità per PegTransfer |
| --- | --- | --- |
| [Brown et al., 2017](https://doi.org/10.1109/TBME.2016.2634861) | Predice cinque domini GEARS da forze, accelerazioni e tempo, usando valutazioni umane. Usa statistiche dei segnali e modelli di regressione/classificazione; non un controllo autonomo. | Misurare le interazioni oltre alla riuscita. I sensori del nostro simulatore differiscono dai loro. |
| [Tang e Cuschieri, 2020 — OCHRA](https://pmc.ncbi.nlm.nih.gov/articles/PMC7093355/) | Analizza errori con conseguenze, quasi incidenti e parti dell’operazione più esposte agli errori. | Registrare quando il peg scivola, quando il donatore rilascia troppo presto e quando serve un recupero. L’adattamento al simulatore va definito e validato. |
| [Laughlin et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12513346/) | Rassegna rubriche, metriche fisiche/cinematiche, valutazioni umane e modelli automatici. | Combinare misure interpretabili e revisione video. Non attribuire automaticamente un voto GEARS alla nostra rete. |
| [Heard et al., 2026](https://www.nature.com/articles/s44484-025-00006-y) | Usa un transformer su gesti contestualizzati per prevedere un esito dopo prostatectomia. Anatomia e funzione aggiungono informazione alla sequenza. | Valutare un gesto insieme a oggetto, braccio, fase e stato della presa. Attenzione del modello e associazioni non provano causalità o sicurezza. |
| [JIGSAWS — dataset ufficiale](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) | Video e cinematica sincronizzati, annotazioni dei gesti e dell’abilità per sutura, annodamento e passaggio dell’ago. | Modello per organizzare il nostro dataset e separare gli esperimenti. Non contiene PegTransfer né etichette direttamente trasferibili di rischio clinico. |
| [Ross et al., 2011 — DAgger](https://proceedings.mlr.press/v15/ross11a.html) | Raccoglie correzioni dell’esperto negli stati visitati dalla policy e le aggrega ai dati di training. | Imparare recuperi sui veri stati di scivolamento o stallo. È una strategia di apprendimento, non una garanzia di safety. |
| [Rajeswaran et al., 2018 — DAPG](https://arxiv.org/abs/1709.10087) | Inizializza con BC e migliora con policy gradient mantenendo un contributo delle dimostrazioni che si riduce nel tempo. | Motiva l’uso delle dimostrazioni durante RL. DAPG usa natural policy gradient: non coincide con il nostro SAC residuale su BC congelata. |
| [Kawaharazuka et al., 2024](https://arxiv.org/abs/2405.03440) | Vincoli geometrici sui port e vincoli di movimento durante le dimostrazioni migliorano l’imitazione. Modello ricorrente con LSTM e immagini compresse. | Vincoli espliciti e dati coerenti aiutano l’apprendimento. Non equivale alla nostra molla che sostiene la presa: i due meccanismi vanno distinti. |

## Punto di partenza verificato

Confronto V5, seed59002, 12 reset identici, tre coppie L5→R2, L6→R1, L4→R3, jitter±0,1mm e yaw±0,5°. Entrambi usano gli stessi aiuti temporanei.

| Misura già presente nei report | BC senza residuo | BC + SAC V5 |
| --- | --- | --- |
| Episodi riusciti | 12/12 | 12/12 |
| Tempo simulato medio, tutti i 12 episodi riusciti | 44,85s | 45,48s |
| Massimo registrato della forza normale su una singola dita | 0,694N | 0,694N |
| Massimo errore finale del foro | 0,217mm | 0,276mm |
| Passi di controllo con recupero dell’apertura PSM1 | 277 | 696 |

Questi dati **non dimostrano un miglioramento di safety con SAC**. I passi di intervento non sono incidenti distinti e non includono tutte le forme di assistenza. Sul seed indipendente62002, SAC ottiene11/12 con un timeout L4→R3 prima del lift stabile; manca ancora una baseline BC sugli stessi reset indipendenti. Checkpoint13 resta il riferimento conservato.

Fonti locali: [baseline](../logs/grasp_v2/bc_sac_residual_v5/baseline12/evaluation.json), [BC+SAC](../logs/grasp_v2/bc_sac_residual_v5/confirmation12/evaluation.json), [seed indipendente](../logs/grasp_v2/bc_sac_recovery_retry_v5/independent12/evaluation.json). Dati e SHA256 sono riepilogati in [safety_baseline_2026-10-14.json](../logs/grasp_v2/safety_baseline_2026-10-14.json).

## Metriche proposte

Le definizioni seguenti sono nostre proposte ispirate alla letteratura. Non sono soglie cliniche validate.

| Aspetto | Misure iniziali | Interpretazione |
| --- | --- | --- |
| Intensità del contatto | Picco e95° percentile delle forze per dito; carico di compressione calcolato dalle norme dei contatti | Evitare sovraccarichi senza confondere una presa debole con una presa sicura. Le forze opposte non vanno sommate vettorialmente, perché potrebbero annullarsi. |
| Esposizione al contatto | Durata sopra una soglia dichiarata e integrale dell’eccesso: ∫max(F−F_lim,0)dt | Distinguere un urto breve da una compressione prolungata. Esplicitare se il limite è per dito o per presa. |
| Stabilità | Scivolamento e rotazione peg–strumento, perdita involontaria dei contatti opposti, cadute | Misurare il rischio di perdere il peg durante lift e scambio. Escludere il rilascio intenzionale dal conteggio delle perdite. |
| Collisioni e distanze | Contatti indesiderati con board/post, urti fra strumenti, distanza minima dalle geometrie | Non conteggiare come errore il contatto richiesto per afferrare o inserire. Serve nuova telemetria dei contatti. |
| Movimento | Velocità e accelerazione reali, jerk filtrato, lunghezza del percorso, arresti non previsti | Separare fluidità/efficienza dagli eventi fisici di rischio. Le pause richieste per verificare una presa non sono stalli. |
| Deposito | Errore del foro, inclinazione, stabilità dopo rilascio; urti in inserimento | Il peg deve restare nella posizione finale una volta terminata l’assistenza. |
| Dipendenza dagli aiuti | Numero e durata dei recuperi, forza/coppia e lavoro delle molle, interventi della guida di deposito | Una riuscita assistita va dichiarata. Molla di base, recupero aggiuntivo e guida sono tre interventi diversi. |

Per ogni misura: risultati per episodio, per coppia e per fase fisicamente verificata; mediana, dispersione, code e intervalli di confidenza. Riportare i fallimenti e i timeout, senza eliminarli dalle statistiche. Per il tempo, separare il tempo dei successi dai timeout censurati al limite.

I risultati principali saranno **le violazioni dei criteri di safety, la loro gravità e durata**, insieme ai margini rispetto ai limiti e all’assistenza utilizzata. La percentuale di task completati senza eventi avversi resta una misura secondaria, distinta dal semplice success rate. Un punteggio unico potrebbe nascondere un evento grave dietro un buon tempo: inizialmente non lo usiamo. Le soglie vanno definite e validate prima del confronto: quelle attuali del simulatore non costituiscono una certificazione clinica.

## Limiti della misura attuale

- `state['forces']` deriva da `ContactSensorData.force_matrix_w`: forze **normali** filtrate dita–peg. Non misura tutta la forza tangenziale, la coppia o le reazioni sotto la board. Brown misura anche la board e accelerazioni fisiche a3kHz: i valori e le bande di vibrazione non sono direttamente confrontabili.
- Il controllo è a100Hz, la fisica a400Hz; la traccia diagnostica attuale è a1Hz. Un massimo calcolato al controllo può perdere picchi fra due campioni. RMS, durata degli eventi e jerk richiedono nuove registrazioni dense.
- Registrare contatti dopo ogni passo fisico e cinematica almeno al controllo; inizializzare i nuovi sensori prima dell’avvio PhysX e verificarli con un contatto noto. Differenziare pose con filtri dichiarati e la stessa frequenza per tutti i modelli. Non riprodurre le vibrazioni ad alta frequenza di Brown dal video o dal log a1Hz.
- Il peg attuale è un corpo rigido con contatti cedevoli, non una gomma deformabile con modello di danno validato. Le misure sono indicatori di rischio fisico nel simulatore, non previsioni di lesione dei tessuti.
- Il limite di1N del progetto è una soglia ingegneristica sul contatto registrato, non una soglia clinica tratta dai paper. La molla ha un limite proprio di0,4N: non equivale a un limite di0,4N della forza totale delle pinze.

## Sequenza di lavoro proposta

1. **Congelare il protocollo.** Elenco di eventi, unità, frequenze, soglie iniziali, regole per contatti intenzionali e distinzione fra errore/recupero/timeout. Stesso criterio per tutti i checkpoint; fissarlo prima di ottimizzare la policy.
2. **Aggiungere un valutatore esterno.** Registra segnali fisici, interventi e video sincronizzato senza fornire comandi alla policy. Prima verificare che la strumentazione non cambi il comportamento su un piccolo confronto identico.
3. **Misurare BC contro BC+SAC.** Primo confronto sul seed62002 già problematico. Poi almeno100 episodi per configurazione, sulle tre coppie e più seed tenuti fuori dal training. Usare reset identici e uguale assistenza, attrito e velocità. Conservare anche i casi non riusciti.
4. **Verificare le etichette sul video.** Rivedere un campione riuscito, fallito e con recupero insieme al professore; se possibile due valutatori. La classificazione degli eventi deve corrispondere alla fisica. Per un voto ispirato a GEARS servono etichette e verifica dell’accordo, non basta una formula inventata.
5. **Allenare sulla criticità misurata.** DAgger per recuperi di presa; BC+SAC con costi separati per sovraccarico/collisione/perdita di presa e ancoraggio alle dimostrazioni. All’inizio SAC agisce sugli spostamenti liberi; contatto solo dopo test specifici. Una penalità nella reward non costituisce una garanzia di safety.
6. **Allargare un solo fattore alla volta.** Prima pose iniziali; poi velocità; poi riduzione degli aiuti. Conservare un test finale separato e riportare sempre il compromesso fra riuscita, tempo, rischio e interventi. Nessuna promozione basata su un solo video o su una media favorevole.

Obiettivo della prima fase: una tabella confrontabile **modello × scenario × fase**, un registro degli eventi con timestamp e un video che permetta di controllarli. Questo rende il risultato discutibile con il professore prima di introdurre un classificatore di safety più complesso.
