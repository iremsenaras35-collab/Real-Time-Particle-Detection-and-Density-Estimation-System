import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd

OUTPUT_DIR = "cikti_kayitlari"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_next_number(prefix, extension):
    files = os.listdir(OUTPUT_DIR)
    nums = []
    for f in files:
        if f.startswith(prefix) and f.endswith(extension):
            try:
                num = int(f.split("_")[1])
                nums.append(num)
            except:
                pass
    return max(nums) + 1 if nums else 1

USE_ROI = True
ROI_Y_START = 130
ROI_HEIGHT = 100
MIN_PIXEL_AREA = 2
MAX_PIXEL_AREA = 1500
BG_FRAMES = 30

LOWER_RED1 = np.array([0, 80, 80])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 80, 80])
UPPER_RED2 = np.array([180, 255, 255])

DIFF_THRESHOLD = 10
MOTION_DIFF_THRESHOLD = 12
PIXELS_PER_CM = 20
CROP_EACH_SIDE_CM = 8.0
CROP_EACH_SIDE_PX = int(CROP_EACH_SIDE_CM * PIXELS_PER_CM)
FULL_BEAM_THRESHOLD = 0.75
MAX_PARTICLE_COUNT = 9999

BEAM_DIAMETER_MM = 3.38
BEAM_RADIUS_MM = BEAM_DIAMETER_MM / 2.0
BEAM_LENGTH_MM = 41.0
BEAM_VOLUME_MM3 = math.pi * (BEAM_RADIUS_MM ** 2) * BEAM_LENGTH_MM
BEAM_VOLUME_CM3 = BEAM_VOLUME_MM3 / 1000.0
print(f"[INFO] Lazer hüzme hacmi: {BEAM_VOLUME_CM3:.6f} cm³")

run_id = time.strftime("%Y%m%d_%H%M%S")
video_num = get_next_number("video_", ".mp4")
VIDEO_FILENAME = os.path.join(OUTPUT_DIR, f"video_{video_num}_{run_id}.mp4")

DUST_DATA_FILE = os.path.join(OUTPUT_DIR, "dustkayit.csv")
EXCEL_FULL = os.path.join(OUTPUT_DIR, "analysis_full.xlsx")
TIME_WINDOW_SECONDS = 10
Y_MAX = 50
DENSITY_Y_MAX = 50

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -7)
prev_gray = None
prev2_gray = None

print("Çıkış için 'q' tuşuna basın.")
print("Kalibrasyon başlıyor…")
bg_masks = []
bg_mask = None
calibrated = False
video_writer = None
VIDEO_FPS = 20.0

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
times, dust_counts, densities, coverage_ratios = [], [], [], []
total_dust = 0
start_time = time.time()
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if USE_ROI:
        frame_roi = frame[ROI_Y_START:ROI_Y_START + ROI_HEIGHT, :]
    else:
        frame_roi = frame.copy()
        
    h, w, _ = frame_roi.shape
    crop_px = min(CROP_EACH_SIDE_PX, w // 2 - 1)
    frame_roi = frame_roi[:, crop_px:w-crop_px]
    display_frame = frame_roi.copy()
    
    if video_writer is None:
        out_h, out_w, _ = display_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(VIDEO_FILENAME, fourcc, VIDEO_FPS, (out_w, out_h))
        
    hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1) + cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    red_mask = cv2.dilate(red_mask, np.ones((2, 2), np.uint8), iterations=1)
    
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        prev_gray = gray.copy()
        prev2_gray = gray.copy()
        
    total_pixels = frame_roi.size // 3
    red_ratio = cv2.countNonZero(red_mask) / total_pixels if total_pixels > 0 else 0
    
    if not calibrated:
        bg_masks.append(red_mask.copy())
        if len(bg_masks) >= BG_FRAMES:
            bg_mask = np.median(np.stack(bg_masks, axis=2), axis=2).astype(np.uint8)
            calibrated = True
            print("[INFO] Kalibrasyon tamam!")
        video_writer.write(display_frame)
        cv2.imshow("Analiz Ekrani", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
        
    diff_bg = cv2.absdiff(red_mask, bg_mask)
    _, dust_bg_mask = cv2.threshold(diff_bg, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    diff1 = cv2.absdiff(gray, prev_gray)
    diff2 = cv2.absdiff(prev_gray, prev2_gray)
    motion_mask = cv2.bitwise_and(diff1, diff2)
    _, motion_mask = cv2.threshold(motion_mask, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, np.ones((2,2), np.uint8), iterations=1)
    
    prev2_gray = prev_gray.copy()
    prev_gray = gray.copy()
    
    _, hot_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    hot_mask = cv2.dilate(hot_mask, np.ones((2,2), np.uint8), iterations=1)
    white_ratio = cv2.countNonZero(hot_mask) / total_pixels if total_pixels > 0 else 0
    
    combined = dust_bg_mask | motion_mask | hot_mask
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    combined = cv2.dilate(combined, np.ones((2,2), np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    particle_count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_PIXEL_AREA <= area <= MAX_PIXEL_AREA:
            particle_count += 1
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cv2.circle(display_frame, (x + w_box // 2, y + h_box // 2), 2, (0, 255, 0), -1)
            
    if red_ratio > FULL_BEAM_THRESHOLD or white_ratio > FULL_BEAM_THRESHOLD:
        particle_count = MAX_PARTICLE_COUNT
        
    elapsed = time.time() - start_time
    density_val = particle_count / BEAM_VOLUME_CM3
    total_dust += particle_count
    
    times.append(elapsed)
    dust_counts.append(particle_count)
    densities.append(density_val)
    coverage_ratios.append(red_ratio * 100)
    
    frame_counter += 1
    
    if frame_counter % 5 == 0:
        ax1.clear()
        ax2.clear()
        tmax = times[-1]
        tmin = max(0, tmax - TIME_WINDOW_SECONDS)
        idx = [i for i, t in enumerate(times) if t >= tmin]
        if idx:
            s = idx[0]
            ax1.plot(times[s:], dust_counts[s:], color='blue')
            ax2.plot(times[s:], densities[s:], color='green')
        ax1.set_xlim(tmin, tmin + TIME_WINDOW_SECONDS)
        ax1.set_ylim(0, Y_MAX)
        ax2.set_ylim(0, DENSITY_Y_MAX)
        ax1.set_ylabel("Toz Sayısı")
        ax2.set_ylabel("Yoğunluk")
        ax1.grid(True)
        ax2.grid(True)
        plt.pause(0.001)
        
    cv2.putText(display_frame, f"Toz: {particle_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Yogunluk: {density_val:.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Toplam Toz: {total_dust}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    video_writer.write(display_frame)
    cv2.imshow("Analiz Ekrani", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()

mean_density = np.mean(densities) if len(densities) > 0 else 0

video_base = os.path.splitext(os.path.basename(VIDEO_FILENAME))[0]

dust_graph = os.path.join(OUTPUT_DIR, f"{video_base}_dust_plot.png")
plt.figure()
plt.plot(times, dust_counts, marker='o')
plt.xlabel("Zaman (s)")
plt.ylabel("Toz Sayısı")
plt.title("Toz Sayısı - Zaman")
plt.grid(True)
plt.text(0.5, -0.2, f"Toplam Toz: {total_dust}", transform=plt.gca().transAxes, ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(dust_graph, dpi=200)
plt.close()

density_graph = os.path.join(OUTPUT_DIR, f"{video_base}_density_plot.png")
plt.figure()
plt.plot(times, densities, marker='o', color='green')
plt.xlabel("Zaman (s)")
plt.ylabel("Yoğunluk (parç/cm3)")
plt.title("Yoğunluk - Zaman")
plt.grid(True)
plt.text(0.5, -0.2, f"Ortalama Yoğunluk: {mean_density:.3f}", transform=plt.gca().transAxes, ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(density_graph, dpi=200)
plt.close()

summary_graph = os.path.join(OUTPUT_DIR, f"{video_base}_summary.png")
plt.figure()
plt.bar(["Toplam Toz", "Ort. Yoğunluk"], [total_dust, mean_density])
plt.title("Ölçüm Özeti")
plt.ylabel("Değer")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(summary_graph, dpi=200)
plt.close()

values_csv = os.path.join(OUTPUT_DIR, f"{video_base}_values.csv")
with open(values_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_s", "dust", "density", "coverage_%"])
    for t, c, d, cov in zip(times, dust_counts, densities, coverage_ratios):
        w.writerow([t, c, d, cov])
print("[INFO] Tüm grafikler ve veri dosyası oluşturuldu.")

header = ["run_id", "time_s", "dust_count", "density_parc_cm3", "coverage_percent"]
exist = os.path.isfile(DUST_DATA_FILE)
with open(DUST_DATA_FILE, "a", newline="") as f:
    w = csv.writer(f)
    if not exist:
        w.writerow(header)
    for t, c, d, cov in zip(times, dust_counts, densities, coverage_ratios):
        w.writerow([run_id, t, c, d, cov])

csv_num = get_next_number("dustkayit_", ".csv")
sep_file = os.path.join(OUTPUT_DIR, f"dustkayit_{csv_num}.csv")
with open(sep_file, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for t, c, d, cov in zip(times, dust_counts, densities, coverage_ratios):
        w.writerow([run_id, t, c, d, cov])

df_full = pd.DataFrame({
    "time_s": times,
    "dust_count": dust_counts,
    "density_parc_cm3": densities,
    "coverage_percent": coverage_ratios
})

sheet = f"run_{run_id}"
mode = "a" if os.path.isfile(EXCEL_FULL) else "w"

with pd.ExcelWriter(EXCEL_FULL, engine="openpyxl", mode=mode, if_sheet_exists="new") as w:
    df_full.to_excel(w, sheet_name=sheet+"_all", index=False)
    pd.DataFrame(times, columns=["time_s"]).to_excel(w, sheet_name=sheet+"_times", index=False)
    pd.DataFrame(dust_counts, columns=["dust_count"]).to_excel(w, sheet_name=sheet+"_counts", index=False)
    pd.DataFrame(densities, columns=["density_parc_cm3"]).to_excel(w, sheet_name=sheet+"_density", index=False)
    pd.DataFrame(coverage_ratios, columns=["coverage_percent"]).to_excel(w, sheet_name=sheet+"_coverage", index=False)

print("[DONE] Tüm kayıtlar başarıyla tamamlandı!")