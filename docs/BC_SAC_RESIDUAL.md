# BC + SAC a partire da checkpoint13

La BC già riuscita rimane congelata. SAC impara una correzione da sommare ai suoi comandi. Non carichiamo il file BC come se fosse un modello SAC: sono reti con strutture diverse. L'esperto serve alla raccolta delle dimostrazioni BC; non viene chiamato durante questo nuovo training o l'inference.

Il primo curriculum permette solo correzioni di traslazione su bracci senza contatto. La correzione massima è 0,15 mm del riferimento richiesto, con una crescita graduale fra 6 e 25 mm di distanza dal volume del peg. Rotazione e pinze restano comandate dalla BC. Non è ancora un training SAC della presa: prima verifichiamo che SAC preservi la sequenza completa.

SAC osserva 249 valori: i 110 dati del wrapper, il comando BC, la scelta e gli obiettivi memorizzati dalla BC, oltre alle memorie del controllo e dei vincoli. Gli otto valori aggiunti al primo prototipo241 descrivono apertura stabile, validità della memoria e stato del recupero della molla. Produce 14 correzioni continue; nel primo livello sono attive solo le sei componenti di traslazione. La BC continua a leggere i suoi originali 98 ingressi. Il replay contiene le correzioni SAC, non i comandi finali, e distingue i timeout dai fallimenti. I terminali conservano lo stato precedente al reset.

| Parametro iniziale | Valore |
| --- | --- |
| Actor e due critic | MLP 256 + 256 |
| Learning rate | 0,00003 |
| Batch / replay | 512 / 100.000 transizioni, su CUDA |
| Gamma / tau | 0,9995 / 0,005 |
| Entropia | 0,00001, fissa |
| Actor iniziale | Correzione media zero; deviazione circa 0,0067 |
| Regolarizzazione | Penalizza correzioni e deviazione dall'esplorazione iniziale |
| Critic iniziale | 512 aggiornamenti su vere traiettorie BC nella stessa fisica assistita |
| Primo pilot | 12.000 transizioni online, sei ambienti |

La reward mantiene i criteri fisici precedenti: avvicinamento, presa, lift, scambio, deposito libero stabile e penalità per tempo/forza/fallimento. Aggiunge solo un piccolo costo della correzione. Molla di presa, supporto verticale, guida al deposito e controllo delle forze restano esplicitamente attivi. `native_full_policy_verified` rimane falso.

Il workflow verifica BC a correzione zero su tre episodi fissi e 12 con piccole variazioni. Poi raccoglie traiettorie BC, allena SAC, confronta 12 episodi con gli stessi reset e registra un episodio frontale del candidato. Blocca il training se la baseline fallisce. Un candidato non diventa automaticamente un nuovo checkpoint certificato: prima dell'aumento della randomizzazione o della riduzione degli aiuti serve una conferma con seed indipendente.

La V1 si è fermata nella raccolta a residuo zero:5/6, senza aggiornamenti SAC. La diagnosi ha riprodotto lo stallo L4→R3: PSM1 perde pressione mentre PSM2 arriva al waypoint esterno; la chiusura aggiuntiva peggiora la presa e la rete non passa al reach. Limite di chiusura generico, filtri delle collisioni e molla sempre più rigida sono stati scartati: non risolvono o alterano altre prese.

La V4 aggiunge un recupero temporaneo misurato. Dopo un lift già stabile, se PSM1 è ancora il solo possessore e la pressione di una sua dita scende sotto0,25N, la molla passa da300 a900N/m e la pinza può tornare gradualmente all'ultima apertura con contatti forti opposti. Il recupero termina al primo contatto di PSM2; resta300N/m sul ricevente. Sono invariati forza massima0,4N, coppia0,004Nm, collisioni originali e criteri di successo. Nessun waypoint o passaggio di fase viene imposto. La prova sul seed fallito61001 completa **6/6**, con interventi sulla pinza soltanto nell'episodio prima bloccato. È assistenza fisica e di apertura, esplicitamente dichiarata, da ridurre in seguito.

```bash
conda activate PegTransfer
bash peg_grasp_v2.sh residual-tests
# Workflow V4 già rilanciato; non avviarne una seconda copia:
watch -n 5 cat logs/grasp_v2/bc_sac_residual_v4/status.json

# Dopo il completamento, inference del candidato con configurazione archiviata:
bash peg_grasp_v2.sh residual-play \
  --checkpoint logs/grasp_v2/bc_sac_residual_v4/training/candidate.zip \
  --num_envs 1 --episodes 3 --seed 59003 --realtime

# Scaricare il video dal terminale del Mac, fuori dalla sessione SSH:
scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/bc_sac_residual_v4.mp4 ~/Downloads/
```

Ogni esecuzione richiede una cartella nuova. Pesi BC, sorgenti e configurazioni sono copiati per l'esperimento; SHA256 verificati prima del replay. Checkpoint13 e i suoi video rimangono integri e sempre riproducibili con `bash peg_grasp_v2.sh checkpoint13-play`.


BC+SAC V4 conclusa: 12.000 transizioni online su GPU. Baseline BC assistita 12/12; candidato BC+SAC 11/12, un timeout e zero fallimenti. Video frontale completo 1/1 riuscito, 41,24 s, con SHA256 verificati. Il candidato resta sperimentale e non sostituisce checkpoint13. Prima di altri training va diagnosticato il timeout.


V5 in corso: prima verifica il recupero anticipato della pressione durante la misura del lift con gli stessi pesi V4 e12 reset. La presa deve essere già misurata e il peg sollevato sopra43mm; il test di lift di2s resta invariato. Dopo il gate12/12 e il video, il workflow esegue un nuovo training BC+SAC e la conferma su seed indipendente.

```bash
watch -n 5 cat logs/grasp_v2/bc_sac_recovery_retry_v5/status.json
# Durante il successivo training:
watch -n 5 cat logs/grasp_v2/bc_sac_residual_v5/status.json
```


V5 conclusa e verificata il14 ottobre: il recupero anticipato risolve il precedente episodio fallito (12/12 sugli stessi reset). Nuovo training BC+SAC12000 transizioni concluso; confronto con baseline12/12 superato. Sul seed indipendente62002:11/12, un timeout dell’episodio8 L4→R3 dopo grasp ma prima del lift stabile, zero fallimenti fisici. Nessuna promozione: checkpoint13 resta il riferimento. Prima di attribuire il limite a SAC serve il confronto BC a residuo zero sullo stesso seed indipendente. Video candidato `logs/grasp_v2/videos/bc_sac_residual_v5.mp4`:1/1 riuscito,42s,1050frame1280×720, intera decodifica e SHA256 verificati, visuale frontale controllata. Nessun training o valutazione residua attivo.
