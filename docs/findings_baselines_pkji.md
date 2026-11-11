# Temuan: Baseline Lampu Lalu Lintas dan Analisis PKJI 2023

Proyek: Lampu Lalu Lintas Adaptif dengan Deep Reinforcement Learning (COMP6826001)
Tanggal: 2026-10-05
Tahap: Minggu 3-4 (baseline), dikerjakan setelah rekap Minggu 1-2 (`docs/recap_week_01-02.md`)

## Ringkasan

- Empat controller dibandingkan pada split **validasi** (seed 60-79, skenario balanced,
  unbalanced, surge, shift; 80 episode per controller, kendaraan identik).
- **Timer tetap PKJI 2023** (hijau 15/17 s, siklus 40 s) adalah timer tetap terbaik:
  rata-rata waktu tunggu 27,2 s. Ini dipakai sebagai baseline utama "cara Indonesia saat ini".
- **Queue rule** (aturan antrean sederhana) jauh lebih baik dari semua timer tetap:
  13,1 s. Ini menjadi tantangan utama bagi DQN.
- Target DQN dari rencana proyek (minimal 10% lebih baik dari timer tetap) berarti
  rata-rata tunggu **di bawah sekitar 24,5 s**.
- Arus jenuh menurut PKJI (1.798 SMP/jam) hampir dua kali arus jenuh yang terukur di
  SUMO (931 kend/jam hijau). PKJI sendiri menjelaskan penyebabnya: perilaku pengemudi
  Indonesia berbeda dengan model Barat yang dipakai SUMO. Ini menjadi keterbatasan utama.

## 1. Empat controller yang dibandingkan

| Controller | Cara kerja | Pengaturan |
|---|---|---|
| Fixed 30/30 | Timer tetap awal, hanya untuk uji simulasi | Hijau 30/30 s, siklus 68 s |
| Fixed Webster terkalibrasi | Rumus Webster (MKJI 1997 / PKJI) dengan arus jenuh **diukur di SUMO** | Hijau 23/24 s, siklus 55 s |
| **Fixed PKJI 2023** | Prosedur PKJI 2023 lengkap dengan arus jenuh dari **rumus dan grafik PKJI** | **Hijau 15/17 s, siklus 40 s** |
| Queue rule | Setelah hijau minimum, ganti fase jika antrean sisi merah lebih panjang dari sisi hijau lebih dari *margin* kendaraan | Margin 1, keputusan tiap 5 s |

Aturan keselamatan sama untuk semua controller (diterapkan oleh `src/traffic_rl/signal.py`):
hijau minimum 10 s, hijau maksimum 60 s, kuning 3 s, merah semua 1 s, transisi tidak bisa
dipotong, setiap intervensi dicatat.

Semua timer tetap dirancang **sekali** dari permintaan rata-rata data validasi
(312 kend/jam untuk NS, 333 kend/jam untuk EW, pada pendekat kritis), lalu dibekukan dan
dipakai tanpa perubahan untuk semua skenario dan seed. Data test tidak pernah dipakai
untuk desain atau tuning.

## 2. Hasil perbandingan (validasi, 80 episode)

| Metrik | Fixed 30/30 | Fixed Webster terkalibrasi | **Fixed PKJI 2023** | Queue rule |
|---|---|---|---|---|
| **Rata-rata waktu tunggu** | 37,5 s | 34,1 s | **27,2 s** | 13,1 s |
| Waktu tunggu maksimum | 193 s | 165 s | 124 s | 78 s |
| Rata-rata tunggu pendekat terburuk | 69,5 s | 61,2 s | 46,3 s | 18,3 s |
| Rata-rata antrean total | 10,8 | 9,9 | 8,0 | 4,0 |
| Throughput (kend/episode) | 593 | 593 | 593 | 593 |
| Kendaraan tak terlayani | 0 | 0 | 0 | 0 |
| Pergantian fase per episode | 58 | 71 | 96 | 80 |

Rata-rata waktu tunggu per skenario:

| Skenario | Fixed 30/30 | Fixed Webster terkalibrasi | **Fixed PKJI 2023** | Queue rule |
|---|---|---|---|---|
| balanced | 20,5 s | 18,9 s | 14,9 s | 12,4 s |
| unbalanced | 53,3 s | 47,3 s | 37,9 s | 12,1 s |
| surge | 29,0 s | 28,0 s | 23,5 s | 15,3 s |
| shift | 47,2 s | 42,4 s | 32,5 s | 12,5 s |

Data lengkap per episode: `results/eval/<controller>_validation.csv` dan `.json`.

## 3. Arus jenuh terukur di SUMO

Script: `scripts/measure_saturation.py`, hasil: `configs/saturation.json`.

Metode: pendekat Utara dibuat jenuh berlebih (1.800 kend/jam) sehingga selalu ada antrean;
pendekat lain 300 kend/jam sehingga kendaraan belok kanan dari Utara menghadapi arus
berlawanan yang wajar. Timer tetap 40/20 selama 1 jam. Arus jenuh = kendaraan yang
meninggalkan pendekat Utara selama hijau NS (saat antrean minimal 3 kendaraan) per jam hijau.

| Kondisi | Arus jenuh |
|---|---|
| Hanya lurus | 1.741 kend/jam hijau |
| Campuran belok (60% lurus, 20% kiri, 20% kanan) | **931 kend/jam hijau** |
| Pembanding: J₀ = 600 × LE (tipe P, PKJI) | 1.920 SMP/jam |

Nilai lurus saja (1.741) mendekati 600 × LE (1.920), artinya kendaraan di SUMO berperilaku
wajar. Dengan belok kanan, kapasitas turun hampir setengah karena kendaraan belok kanan
menunggu celah arus berlawanan dan menghalangi lajur yang dipakai bersama.

## 4. Desain timer PKJI 2023

Sumber: Direktorat Jenderal Bina Marga (2023), *Pedoman Kapasitas Jalan Indonesia*
(No. 09/P/BM/2023), Bab 5 Kapasitas Simpang APILL dan Lampiran 12. Nomor halaman di bawah
adalah nomor halaman dokumen ("x dari 326").

### 4.1 Arus jenuh (Persamaan 5-4, hal. 108)

J = J₀ × FHS × FUK × FG × FP × FBKi × FBKa

| Parameter | Nilai | Dasar | Sumber |
|---|---|---|---|
| Tipe pendekat | Terlawan (O) | Belok kanan mendapat hijau bersamaan dengan arus lurus dari arah berlawanan | Gambar 5-4, hal. 106 |
| LE | 3,2 m | Satu lajur per arah | |
| qBKa, qBKa,O | ±65 SMP/jam | 20% dari arus pendekat rata-rata | |
| **J₀** | **1.713 SMP/jam** | Dibaca dari grafik untuk LE 3,0 dan 4,0 m, kurva qBKa,O 50 dan 100, lalu interpolasi linear | Gambar 12-2, hal. 257 |
| FHS | 1,00 | Akses terbatas, terlawan, rasio kendaraan tak bermotor 0 | Tabel 12-3, hal. 270 |
| FUK | 1,05 | Penduduk > 3 juta jiwa (Jakarta) | Tabel 12-1, hal. 269 |
| FG | 1,00 | Datar | Gambar 12-10, hal. 265 |
| FP | 1,00 | Tidak ada parkir | Pers. 5-5, Gambar 12-11 |
| FBKa | 1,00 | Gambar 12-12 hanya untuk tipe P; pada tipe O pengaruh belok kanan sudah termasuk dalam J₀ | Gambar 12-12, hal. 266 |
| FBKi | 1,00 | Gambar 12-13 hanya untuk tipe P | Gambar 12-13, hal. 266 |
| **J** | **1.798 SMP/jam** | | |

Satuan: semua kendaraan di simulasi adalah mobil penumpang (EMP = 1,00, Tabel 5-2 hal. 114),
jadi kend/jam = SMP/jam.

Pembacaan grafik J₀ dilakukan secara visual (ketelitian sekitar ±20 SMP/jam). Hasil desain
tidak sensitif terhadap ketelitian ini karena waktu siklus dibatasi ke nilai minimum 40 s.

### 4.2 Waktu isyarat

| Langkah | Nilai | Sumber |
|---|---|---|
| Waktu kuning | 3 s (biasa dipakai di kota-kota Indonesia) | hal. 112 |
| Waktu antar hijau | 4 s per fase untuk simpang kecil (lebar jalan rata-rata 6 sampai < 10 m; jalan kita 6,4 m) | Tabel 5-1, hal. 103 |
| wHH | Σ(wMS + wK) = 2 × (1 + 3) = 8 s | Pers. 5-10, hal. 112 |
| Rq/J | NS 0,173; EW 0,185 | Pers. 5-8, hal. 110 |
| ΣRq/J kritis | 0,359 | |
| Waktu siklus | s = (1,5 × 8 + 5) / (1 − 0,359) = 26,5 s | Pers. 5-11, hal. 112 (Webster) |
| Siklus yang layak | 40 sampai 80 s untuk dua fase; nilai rendah untuk pendekat < 10 m | Tabel 12-2, hal. 269; hal. 124 |
| **Siklus dipakai** | **40 s** | |
| Waktu hijau | wHi = (s − wHH) × Rq/J kritis / ΣRq/J = **15 s (NS) / 17 s (EW)** | Pers. 5-12, hal. 113 |
| Derajat kejenuhan | DJ = 0,46 (NS) / 0,44 (EW), memenuhi kriteria DJ ≤ 0,85 | Pers. 5-14; hal. 104 |

Pengaturan kuning dan merah semua di simulasi sudah sesuai PKJI, sehingga aturan keselamatan
untuk semua controller tidak perlu diubah.

## 5. Tuning queue rule

Script: `scripts/tune_queue_rule.py`, hasil: `configs/queue_rule.json`. Aturan pemilihan
ditetapkan sebelum melihat hasil: rata-rata waktu tunggu terendah pada validasi, di antara
margin yang jumlah kendaraan tak terlayaninya tidak lebih buruk dari yang terbaik.

| Margin (kendaraan) | 0 | **1** | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Rata-rata tunggu | 13,3 s | **13,1 s** | 15,0 s | 17,2 s | 19,5 s | 21,2 s | 23,4 s |
| Pergantian fase per episode | 100 | 80 | 64 | 54 | 46 | 40 | 37 |

Margin kecil bereaksi cepat tetapi lebih sering mengganti fase (setiap pergantian membuang
4 s untuk kuning dan merah semua). Margin 1 memberi keseimbangan terbaik.

## 6. Temuan dan interpretasi

1. **Metode PKJI menghasilkan timer tetap terbaik.** Siklus yang lebih pendek (40 s)
   membuat kendaraan lebih cepat mendapat giliran hijau dibanding 55 s atau 68 s.
   PKJI juga mencatat bahwa siklus yang terlalu panjang meningkatkan tundaan (hal. 113).
2. **Timer tetap tidak bisa mengikuti perubahan arus.** Pada skenario unbalanced dan shift,
   sisi yang ramai membutuhkan hijau lebih panjang daripada bagian tetapnya, sehingga antrean
   menumpuk (37,9 s dan 32,5 s untuk PKJI, dibanding sekitar 12 s untuk queue rule).
   Di sinilah controller adaptif unggul.
3. **Queue rule sangat kuat untuk simpang ini.** Rata-rata tunggu 52% lebih rendah dari
   timer PKJI. Karena itu queue rule dipertahankan sebagai baseline pendukung: jika DQN
   hanya mengalahkan timer PKJI tetapi tidak mengalahkan queue rule, maka peningkatannya
   berasal dari kemampuan bereaksi terhadap antrean, bukan dari proses pembelajaran.
4. **Perbedaan arus jenuh PKJI dan SUMO.** PKJI memperkirakan J = 1.798 SMP/jam, sedangkan
   SUMO hanya mencapai 931 kend/jam hijau pada pendekat yang sama. PKJI menjelaskan
   (catatan hal. 109-110) bahwa pengemudi Indonesia yang belok kanan sering menerobos arus
   berlawanan dan mengabaikan aturan hak jalan, sehingga model kapasitas Barat berbasis
   *gap acceptance* tidak dapat diterapkan dan kapasitas di Indonesia berbeda. SUMO memakai
   model *gap acceptance* tersebut: kendaraan belok kanan menunggu celah. Akibatnya simulasi
   lebih konservatif daripada perilaku lalu lintas Indonesia yang sebenarnya, dan desain PKJI
   yang di atas kertas aman (DJ 0,46) tetap menghasilkan antrean nyata di simulasi.
5. **Throughput sama untuk semua controller (593 kendaraan) dan tidak ada kendaraan tak
   terlayani** pada tingkat permintaan validasi. Perbedaan antar controller hanya terlihat
   pada waktu tunggu dan antrean. Skenario stress (hanya di test) akan menguji kondisi mendekati
   kapasitas.

## 7. Keterbatasan

- Perilaku belok kanan di SUMO (*gap acceptance*) berbeda dari perilaku Indonesia yang
  dijelaskan PKJI. Hasil hanya berlaku untuk model simulator ini.
- Simpang kecil: satu lajur per arah, dua fase, tanpa lajur belok terpisah, tanpa pejalan kaki,
  sepeda motor, atau kendaraan berat. Simpang besar di Jakarta umumnya 3 sampai 4 fase dengan
  siklus 120 s atau lebih.
- J₀ tipe O dibaca dari grafik secara visual.
- Desain PKJI memakai permintaan rata-rata validasi. Perencana di lapangan memakai arus jam
  puncak (qJP = LHRT × K, Pers. 5-13), yang biasanya lebih tinggi dan menghasilkan siklus
  lebih panjang.

## 8. Keputusan

| Keputusan | Status |
|---|---|
| Baseline utama timer tetap | **Fixed PKJI 2023 (15/17 s, siklus 40 s)** |
| Baseline pendukung | Queue rule, margin 1 |
| Fixed 30/30 dan Webster terkalibrasi | Hanya sebagai catatan; tidak wajib masuk laporan |
| Target DQN | Rata-rata tunggu ≥ 10% lebih rendah dari Fixed PKJI (di bawah ±24,5 s), tanpa penurunan throughput atau kenaikan backlog; perbandingan dengan queue rule tetap dilaporkan |
| Program lampu default di SUMO | Masih Webster terkalibrasi (23/24 s); belum diganti ke PKJI |

## 9. File terkait

| File | Isi |
|---|---|
| `src/traffic_rl/signal.py` | Pengelola lampu dengan aturan keselamatan dan pencatatan intervensi |
| `src/traffic_rl/controllers.py` | Timer tetap dan queue rule |
| `src/traffic_rl/episode.py` | Menjalankan satu episode untuk controller mana pun |
| `src/traffic_rl/evaluation.py` | Evaluasi satu split dan agregasi hasil |
| `scripts/measure_saturation.py` | Pengukuran arus jenuh di SUMO |
| `scripts/design_mkji.py` | Desain timer Webster (S terukur) dan PKJI 2023 (`--saturation pkji_formula`) |
| `scripts/tune_queue_rule.py` | Pemilihan margin queue rule pada validasi |
| `scripts/evaluate.py` | Evaluasi controller pada satu split (test dikunci dengan `--final`) |
| `configs/saturation.json` | Hasil pengukuran arus jenuh |
| `configs/mkji_timer.json` | Desain Webster terkalibrasi |
| `configs/pkji_factors.json` | Semua faktor PKJI 2023 dengan sumber dan nomor halaman |
| `configs/pkji_timer.json` | Desain timer PKJI 2023 |
| `configs/queue_rule.json` | Margin terpilih dan hasil sweep |
| `results/eval/` | Hasil evaluasi per episode dan agregat |

Perintah untuk mengulang:

```powershell
.\.venv\Scripts\python.exe scripts\measure_saturation.py
.\.venv\Scripts\python.exe scripts\design_mkji.py
.\.venv\Scripts\python.exe scripts\design_mkji.py --saturation pkji_formula
.\.venv\Scripts\python.exe scripts\tune_queue_rule.py
.\.venv\Scripts\python.exe scripts\evaluate.py --controller fixed_pkji
.\.venv\Scripts\python.exe scripts\evaluate.py --controller queue_rule
.\.venv\Scripts\python.exe scripts\run_episode.py --scenario unbalanced --seed 60 --controller fixed_pkji --gui
```

## 10. Referensi

- Direktorat Jenderal Bina Marga. (2023). *Pedoman Kapasitas Jalan Indonesia* (No. 09/P/BM/2023).
  Kementerian Pekerjaan Umum dan Perumahan Rakyat.
- Direktorat Jenderal Bina Marga. (1997). *Manual Kapasitas Jalan Indonesia (MKJI)*.
  Departemen Pekerjaan Umum.
- Webster, F. V. (1958). *Traffic signal settings* (Road Research Technical Paper No. 39).
  Her Majesty's Stationery Office.

## 11. Langkah berikutnya

1. Ganti program lampu default SUMO ke desain PKJI (opsional, untuk tampilan SUMO-GUI).
2. Minggu 5-6: environment RL (Gymnasium) di atas `signal.py` dan `metrics.py` yang sama,
   pilot DQN 10.000 sampai 20.000 langkah, dan proposal Minggu 6.
