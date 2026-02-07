# Devnet-2882
Unsupervised Anomaly Detection on Meraki Telemetry Using Isolation Forests

Acquire real operational telemetry  
Learn “normal” behavior without labels  
core recent data for anomalies  
Interpret and visualize results with statistically meaningful thresholds  
  
[devnet-2882/]  
  
├── README.md  
  
├── data/  
├── raw/                # untouched exports / API fetches  
├── processed/          # cleaned, aggregated, feature-ready  
└── samples/            # small example CSVs for reviewers  
  
├── acquire/  
  ______fetch_wireless_events.py  
  ______fetch_mx_uplink.py  
  ______fetch_location_csv.md  
  ______README.md  
  
├── train/  
  ______ftrain_location_iforest.py  
  ______ftrain_mx_uplink_iforest.py  
  ______ftrain_wireless_assoc_iforest.py  

├── models/  
  ______fiforest_location.joblib  
  ______fiforest_mx_uplink.joblib  
  ______fiforest_wireless_assoc.joblib  
  ______fREADME.md  
  
├── infer/  
  ______flert_eval.py       # ONE canonical evaluator  
  ______fthresholds.py  
    ______fREADME.md  
  
├── plot/  
  ______fplot_location.py  
  ______fplot_mx_uplink.py  
  ______fplot_wireless_assoc.py  
  ______fplot_alert_windows.py  
  
├── notebooks/  
  ______fexploration.ipynb  
  ______fmodel_validation.ipynb  
  
├── requirements.txt  
└── .env.example  
  
