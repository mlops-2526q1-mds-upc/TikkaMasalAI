import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from textwrap import dedent

# ==== Inline CSV only
USE_INLINE = True
CSV_TEXT = dedent("""
timestamp,project_name,run_id,experiment_id,duration,emissions,emissions_rate,cpu_power,gpu_power,ram_power,cpu_energy,gpu_energy,ram_energy,energy_consumed,country_name,country_iso_code,region,cloud_provider,cloud_region,os,python_version,codecarbon_version,cpu_count,cpu_model,gpu_count,gpu_model,longitude,latitude,ram_total_size,tracking_mode,on_cloud,pue
2025-10-09T11:25:18,codecarbon,bc722d50-3edb-4c40-90e8-c00219f885a1,5b0fa12a-3dd7-45bb-9766-cc326314d9f1,116.10263690000284,0.0021808710595257,1.8783992489370354e-05,360.0,26.89220618329357,20.0,0.0114969562599988,0.000394497260042,0.000638687586111,0.0125301411061519,Spain,ESP,catalonia,,,Windows-10-10.0.26100-SP0,3.10.18,3.0.7,16,11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz,1,1 x NVIDIA GeForce RTX 3060 Laptop GPU,2.1181,41.383,31.73318862915039,machine,N,1.0
2025-10-09T12:02:19,codecarbon,1e726540-b723-49cc-a778-b22599ff4894,5b0fa12a-3dd7-45bb-9766-cc326314d9f1,177.6458341999969,0.0033516610264486,1.886709610468716e-05,360.0,10.165537544213835,20.0,0.0175923109099938,0.0006872938831679,0.0009772813111108,0.0192568861042726,Spain,ESP,catalonia,,,Windows-10-10.0.26100-SP0,3.10.18,3.0.7,16,11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz,1,1 x NVIDIA GeForce RTX 3060 Laptop GPU,2.1181,41.383,31.73318862915039,machine,N,1.0
2025-10-09T12:51:54,codecarbon,3ec44a9d-83fb-494c-bdce-318a7efe2bf1,5b0fa12a-3dd7-45bb-9766-cc326314d9f1,1748.0804014000169,0.0105803677786425,6.052563583556471e-06,360.0,10.358400166141587,20.0,0.1727463346399832,0.0064609523909799,0.0095963636166663,0.1888036506476296,France,FRA,île-de-france,,,Windows-10-10.0.26100-SP0,3.10.18,3.0.7,16,11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz,1,1 x NVIDIA GeForce RTX 3060 Laptop GPU,2.3494,48.8558,31.73318862915039,machine,N,1.0
2025-10-09T13:42:16,codecarbon,13122db4-e0d1-480a-a38c-0c17e5ca5da0,5b0fa12a-3dd7-45bb-9766-cc326314d9f1,191.3273438999895,0.0011981281348565692,6.262189765634618e-06,360.0,24.418103723402265,20.0,0.019129278590003383,0.0011883103950919988,0.0010626664594444771,0.021380255444539857,France,FRA,île-de-france,,,Windows-10-10.0.26100-SP0,3.10.18,3.0.7,16,11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz,1,1 x NVIDIA GeForce RTX 3060 Laptop GPU,2.3494,48.8558,31.73318862915039,machine,N,1.0
""")

# ==== Load & sort
df = pd.read_csv(StringIO(CSV_TEXT), parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ==== Label logic
labels, epochs, train_images = [], [], []
for _, r in df.iterrows():
    if r["country_name"] == "Spain" and r["duration"] < 150:
        labels.append("Run 1 (2ep,1k)"); epochs.append(2); train_images.append(1000)
    elif r["country_name"] == "Spain":
        labels.append("Run 2 (3ep,1k)"); epochs.append(3); train_images.append(1000)
    elif r["country_name"] == "France" and r["energy_consumed"] > 0.05:
        labels.append("Run 3 (3ep,10k)"); epochs.append(3); train_images.append(10000)
    else:
        labels.append("Run 4 (3ep,1k)"); epochs.append(3); train_images.append(1000)

df["run_label"] = labels
df["epochs"] = epochs
df["train_images"] = train_images

# ==== Metrics
df["emissions_g"] = df["emissions"] * 1000.0
df["co2_per_epoch_g"] = df["emissions_g"] / df["epochs"]
df["co2_per_image_per_epoch_mg"] = (df["co2_per_epoch_g"] * 1000.0) / df["train_images"]

# ==== Figure A: CO2e per run (g)
plt.figure(figsize=(6, 4))
plt.bar(df["run_label"], df["emissions_g"])
plt.title("CO₂e per run (g)")
plt.ylabel("grams CO₂e")
plt.xticks(rotation=20, ha="right")
for i, v in enumerate(df["emissions_g"]):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# ==== Figure B: CO2e per image per epoch (mg)
plt.figure(figsize=(6, 4))
plt.bar(df["run_label"], df["co2_per_image_per_epoch_mg"])
plt.title("CO₂e per image per epoch (mg)")
plt.ylabel("mg CO₂e")
plt.xticks(rotation=20, ha="right")
for i, v in enumerate(df["co2_per_image_per_epoch_mg"]):
    plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# ==== Two-panel 
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
axes[0].bar(df["run_label"], df["emissions_g"])
axes[0].set_title("CO₂e per run (g)")
axes[0].set_ylabel("grams CO₂e")
axes[0].tick_params(axis="x", rotation=20)
for i, v in enumerate(df["emissions_g"]):
    axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom")

axes[1].bar(df["run_label"], df["co2_per_image_per_epoch_mg"])
axes[1].set_title("CO₂e per image per epoch (mg)")
axes[1].set_ylabel("mg CO₂e")
axes[1].tick_params(axis="x", rotation=20)
for i, v in enumerate(df["co2_per_image_per_epoch_mg"]):
    axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()
