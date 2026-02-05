# Devnet-2882
Unsupervised Anomaly Detection on Meraki Telemetry Using Isolation Forests

Acquire real operational telemetry  
Learn “normal” behavior without labels  
core recent data for anomalies  
Interpret and visualize results with statistically meaningful thresholds  
  
[devnet-2882/]  
  
├── README.md  
  
├── data/  
│   ├── raw/                # untouched exports / API fetches  
│   ├── processed/          # cleaned, aggregated, feature-ready  
│   └── samples/            # small example CSVs for reviewers  
  
├── acquire/  
│   ├── fetch_wireless_events.py  
│   ├── fetch_mx_uplink.py  
│   ├── fetch_location_csv.md  
│   └── README.md  
  
├── train/  
│   ├── train_location_iforest.py  
│   ├── train_mx_uplink_iforest.py  
│   ├── train_wireless_assoc_iforest.py  
│   ├── models/  
│   │   ├── iforest_location.joblib  
│   │   ├── iforest_mx_uplink.joblib  
│   │   └── iforest_wireless_assoc.joblib  
│   └── README.md  
  
├── infer/  
│   ├── alert_eval.py       # ONE canonical evaluator  
│   ├── thresholds.py  
│   └── README.md  
  
├── plot/  
│   ├── plot_location.py  
│   ├── plot_mx_uplink.py  
│   ├── plot_wireless_assoc.py  
│   └── plot_alert_windows.py  
  
├── notebooks/  
│   ├── exploration.ipynb  
│   └── model_validation.ipynb  
  
├── requirements.txt  
└── .env.example  
  
