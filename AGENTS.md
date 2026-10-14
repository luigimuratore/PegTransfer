# Preferenze dell'utente sui checkpoint

Ogni volta che conservi un nuovo checkpoint, registra anche un video di un episodio completo usando i pesi, i sorgenti e la configurazione archiviati di quel checkpoint. Usa la visuale frontale della board e dei PSM. Salva il video con il nome del checkpoint, per esempio `logs/grasp_v2/videos/checkpoint11.mp4`, insieme al report dell'episodio e ai metadati che collegano video e pesi tramite SHA256.

Verifica che il video sia leggibile, che mostri l'intero episodio e che il risultato dichiarato corrisponda al report fisico. Un singolo video riuscito non sostituisce la valutazione su più episodi. Non sovrascrivere un video o un checkpoint conservato: per una nuova registrazione aggiungi un suffisso e mantieni l'originale. Aggiungi al README il percorso e il comando per scaricare il video da SSH.
