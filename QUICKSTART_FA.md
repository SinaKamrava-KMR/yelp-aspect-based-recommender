# راهنمای سریع نسخهٔ بهینه‌شده

محتویات ZIP را در ریشهٔ پروژهٔ فعلی `NLP_HW3` استخراج و جایگزینی فایل‌های کد را
تأیید کنید. پوشه‌های `data`، `.venv` و `results/cache` فعلی را حذف نکنید.

## ۱) فعال‌سازی و نصب

```powershell
Set-Location "D:\Shiraz University\semester2\NLP\Projects\NLP_HW3"
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements-advanced.txt
python -m pytest -v
```

## ۲) اجرای همهٔ بهبودهای CPU-safe

```powershell
python -u .\system.py `
  --raw-dir ".\data\raw" `
  --train-reviews ".\data\processed\train_reviews.csv" `
  --businesses ".\data\processed\businesses.jsonl" `
  --dev-reviews ".\data\processed\dev_reviews.csv" `
  --dev-qrels ".\data\processed\dev_qrels.tsv" `
  --dev-users ".\data\processed\dev_users.csv" `
  --test-users ".\data\raw\test_users.csv" `
  --outdir ".\results" `
  --alpha 0.7 `
  --n-process 1 `
  --batch-size 1000 `
  --min-interactions 5 `
  --threshold 0.3 `
  --seed 42 `
  --resume `
  --run-all-experiments `
  --evidence-limit 200 `
  --progress-every 10000 `
  2>&1 | Tee-Object -FilePath ".\optimized_run.log"
```

این فرمان baseline اجباری را از checkpoint اجرای موفق قبلی می‌خواند و روش‌های
بهینه‌شده را جداگانه اجرا می‌کند. هیچ نقدی حذف یا نمونه‌گیری نمی‌شود.

## ۳) آزمایش Transformer

پس از پایان فرمان قبل، همان فرمان را با گزینه‌های زیر دوباره اجرا کنید:

```powershell
python -m pip install -r .\requirements-transformer.txt
```

```powershell
  --use-transformer `
  --transformer-device auto `
```

Transformer روی CPU بسیار زمان‌بر است؛ GPU مناسب ترجیح دارد. سایر checkpointها
دوباره محاسبه نمی‌شوند.

## خروجی‌های اصلی

- `results/dev_metrics.json`: مقایسهٔ تمام مدل‌ها
- `results/ablation.json`: sweep کامل و مدل منتخب
- `results/aspect_diagnostics.json`: تحلیل پوشش و خطاهای Aspect
- `results/error_analysis.json`: تحلیل کاربران سرد، متوسط و پرتعامـل
- `results/test_recommendations.csv`: خروجی مدل منتخب
- `results/test_recommendations_mandatory.csv`: خروجی baseline اجباری
- `results/trace.jsonl`: trace اجباری
- `results/trace_best.jsonl`: trace مدل منتخب

انتخاب مدل فقط روی `dev` و به‌ترتیب `NDCG@10`، سپس `HR@10` و بعد `RMSE` انجام
می‌شود. دادهٔ test در آموزش یا انتخاب مدل دخالت ندارد.
