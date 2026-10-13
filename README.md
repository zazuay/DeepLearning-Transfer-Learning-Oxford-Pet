# Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

Tugas Take-Home Sesi 05: *Advanced CNNs and Transfer Learning*.

Repositori ini membandingkan tiga paradigma pada ResNet-18 untuk klasifikasi 10 ras kucing dan anjing (Oxford-IIIT Pet, 1.973 gambar): *training from scratch*, *feature extraction*, dan *fine-tuning* sebagian (`layer4`). Eksperimen tambahan mencakup uji kontrol augmentasi, perbandingan dengan EfficientNet-B0 dan MobileNetV3-Small, pengulangan 3 seed, *failure analysis*, dan Grad-CAM.

> Ganti `<username>` pada tautan badge dengan username GitHub kamu.

## Isi repositori

```
tugas05-transfer-learning-oxford-pet/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook.ipynb              # notebook lengkap dengan output sel
├── report/
│   ├── Laporan_Bagian_C.pdf    # laporan analisis (Bagian C)
│   └── Bagian_A_Analisis_Konseptual.pdf
├── results/
│   ├── config.json             # konfigurasi eksperimen (seed, hyperparameter, versi library)
│   ├── results.json            # log metrik mentah per eksperimen
│   ├── results_summary.csv     # tabel ringkasan B1-B5
│   ├── results_multiseed.csv   # hasil per seed (B2, B3, B4)
│   └── analysis_extra.json     # analisis tambahan (pasangan konfusi, uji McNemar, dsb.)
└── figures/
    ├── curves_B1_B2_B3.png     # kurva train vs val loss
    ├── curves_B4.png           # B3 dengan vs tanpa augmentasi
    ├── failure_analysis.png    # 8 kesalahan test set
    ├── crop_check.png          # gambar asli vs hasil Resize + CenterCrop
    └── gradcam_B2_vs_B3.png    # Grad-CAM layer4, B2 vs B3
```

## Dataset dan protokol

- **Data:** Oxford-IIIT Pet (Parkhi et al., 2012), diunduh otomatis lewat `torchvision.datasets.OxfordIIITPet(download=True)`. Subset 10 kelas:
  - Kucing: Abyssinian, Bengal, Egyptian Mau, Birman, Ragdoll.
  - Anjing: American Pit Bull Terrier, Staffordshire Bull Terrier, Beagle, Basset Hound, English Cocker Spaniel.
- **Split:** stratified 70/15/15 menjadi 1.381 / 296 / 296 (train / val / test). Notebook memeriksa bahwa tidak ada *overlap* antar-*split*.
- **Model:** ResNet-18 (bobot `IMAGENET1K_V1`), batch 32, AdamW (*weight decay* 1e-2), CrossEntropyLoss.
- **Early stopping:** memantau val loss, maks. 40 epoch, *patience* 5, bobot terbaik dipulihkan. Pengecualian: B1_ext (maks. 60 epoch, *patience* 10).
- **Input:** `Resize(256)` + `CenterCrop(224)` + normalisasi ImageNet untuk val/test. Augmentasi (RandomResizedCrop, flip horizontal, rotasi 15°, color jitter) hanya untuk train.
- **Backbone beku** di-`.eval()` agar *running statistics* BatchNorm tidak berubah. Notebook memuat pembuktiannya.

| Kode | Kondisi | Deskripsi |
|---|---|---|
| B1 | *From scratch* | Inisialisasi acak, semua parameter dilatih |
| B2 | *Feature extraction* | Backbone beku, hanya `fc` baru yang dilatih |
| B3 | *Fine-tuning* | `layer4` + `fc` dilatih, LR 1e-4 / 1e-3 |
| B4 | Kontrol augmentasi | B3 tanpa augmentasi |
| B5 | Arsitektur pembanding | EfficientNet-B0 (B5a) dan MobileNetV3-Small (B5b) |

## Cara menjalankan

1. Klik badge Colab di atas, atau unggah `notebook.ipynb` ke [Google Colab](https://colab.research.google.com/).
2. Pilih runtime **GPU** (*Runtime → Change runtime type → T4 GPU*).
3. Jalankan **Runtime → Run all**. Dataset (sekitar 800 MB) diunduh otomatis ke folder `data/`.
4. Notebook menulis `results.json`, `results_summary.csv`, `results_multiseed.csv`, `analysis_extra.json`, `config.json`, dan semua gambar ke direktori kerja Colab. Di repositori ini file tersebut dipindahkan ke `results/` dan `figures/`.

Menjalankan di lingkungan lokal:

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Notebook dibuat untuk Colab. Sel instalasi dan unggah/unduh file khusus Colab (`google.colab`) perlu dilewati atau disesuaikan di lingkungan lokal.

Total waktu training eksperimen utama (B1-B5) sekitar 30 menit pada GPU T4 (jumlah kolom waktu pada tabel di bawah). Multi-seed dan pengunduhan dataset menambah waktu.

## Reproduksibilitas

- Seed tetap **42** untuk Python, NumPy, dan PyTorch. Multi-seed memakai seed 42, 43, 44.
- `torch.backends.cudnn.deterministic = True` dan `benchmark = False`.
- Versi: `torch 2.11.0+cu130`, `torchvision 0.26.0+cu130`. Daftar lengkap ada di `requirements.txt` dan `results/config.json`.
- Hasil dapat sedikit berbeda antar-run karena perbedaan GPU atau *worker* DataLoader. Waktu dan latensi paling sensitif.

## Hasil ringkas (seed 42)

| Kondisi | Strategi | Trainable | LR | Epoch (best/jalan) | Waktu (s) | Val acc | Test F1 | Gap |
|---|---|---|---|---|---|---|---|---|
| B1 | *Scratch* (patience 5) | 11.181.642 | 1e-3 | 13/18 | 266,5 | 0,399 | 0,411 | 0,053 |
| B1_ext | *Scratch* (patience 10) | 11.181.642 | 1e-3 | 39/49 | 718,9 | 0,514 | 0,504 | 0,176 |
| B2 | *Feature extraction* | 5.130 | 1e-3 | 14/19 | 261,1 | 0,885 | 0,892 | 0,055 |
| B3 | *Fine-tune* `layer4` + aug | 8.398.858 | 1e-4 / 1e-3 | 4/9 | 125,4 | 0,902 | 0,895 | 0,077 |
| B4 | B3 tanpa augmentasi | 8.398.858 | 1e-4 / 1e-3 | 4/9 | 90,7 | 0,909 | 0,898 | 0,091 |
| B5a | EfficientNet-B0 | 1.142.202 | 1e-4 / 1e-3 | 10/15 | 204,7 | 0,926 | 0,905 | 0,060 |
| B5b | MobileNetV3-Small | 951.642 | 1e-4 / 1e-3 | 4/9 | 119,5 | 0,882 | 0,894 | 0,066 |

*Gap = train acc (tanpa augmentasi) - val acc. Test F1 = Macro F1.*

**Multi-seed (3 seed, rata-rata ± std):**

| Kondisi | Test F1 | Val acc | Gap | Best epoch |
|---|---|---|---|---|
| B2 | 0,8923 ± 0,0056 | 0,8863 ± 0,0052 | 0,0454 ± 0,0088 | 16,7 ± 3,1 |
| B3 | 0,8985 ± 0,0055 | 0,8975 ± 0,0039 | 0,0754 ± 0,0049 | 4,0 ± 1,0 |
| B4 (tanpa aug) | 0,9017 ± 0,0089 | 0,9144 ± 0,0052 | 0,0856 ± 0,0052 | 5,3 ± 1,5 |

**Efisiensi (latensi diukur batch 32, input acak):**

| Model | Total param. | Trainable | Test F1 | Waktu (s) | Latensi (ms/batch) |
|---|---|---|---|---|---|
| ResNet-18 (B3) | 11.181.642 | 8.398.858 | 0,895 | 125,4 | 31,7 |
| EfficientNet-B0 | 4.020.358 | 1.142.202 | 0,905 | 204,7 | 40,6 |
| MobileNetV3-S | 1.528.106 | 951.642 | 0,894 | 119,5 | 8,3 |

## Temuan utama

- Pretrained jauh lebih baik daripada *from scratch* (F1 0,892-0,895 vs 0,411-0,504) pada data sekecil ini.
- B2 dan B3 setara pada Macro F1 (uji McNemar seed 42: p = 1,0). B3 konvergen jauh lebih cepat (sekitar 4 epoch vs sekitar 17), tetapi gap train-val lebih besar.
- Augmentasi tidak menambah akurasi pada protokol ini (B4 tanpa augmentasi: F1 0,9017 vs 0,8985, dalam std).
- MobileNetV3-Small menyamai B3 dengan latensi sekitar 3,8× lebih rendah. Selisih F1 EfficientNet-B0 terhadap B3 (sekitar 1 poin, satu seed) tidak dapat dibedakan dari *noise*.

## Limitasi

- Satu dataset, 10 kelas, domain dekat ImageNet.
- Hanya B2, B3, dan B4 yang diulang 3 seed. B1, B1_ext, dan B5 satu seed.
- Test set 296 gambar (SE akurasi sekitar 1,7 pp), sehingga selisih kecil tidak bermakna.
- Val set dipakai untuk *early stopping* dan pemilihan model *failure analysis*.
- `layer4` dibuka sekaligus, tanpa *gradual unfreezing* atau `lr_scheduler`.
- Hipotesis *crop* pada *failure analysis* hanya berdasar inspeksi visual dan belum diuji dengan prediksi ulang tanpa *crop*.

Pembahasan lengkap ada di `report/Laporan Analisis.pdf`.

## Pernyataan Penggunaan AI

Pernyataan penggunaan AI ada di bagian akhir laporan (`report/Laporan Analisis.pdf`).

## Referensi

- Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and dogs. *CVPR*. DOI: 10.1109/CVPR.2012.6248092.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770-778.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML* (PMLR 97), 6105-6114.
- Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV*, 1314-1324.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.