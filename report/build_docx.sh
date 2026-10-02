#!/usr/bin/env bash
# report/izvestaj.md -> report/izvestaj.docx
# Putanje do slika su relativne prema KosnicaML/, pa pandoc radi odatle.
set -euo pipefail
cd "$(dirname "$0")/.."

./.venv/bin/python -m src.make_report

pandoc report/izvestaj.md \
  --from=markdown \
  --to=docx \
  --output=report/izvestaj.docx \
  --resource-path=.:report \
  --toc --toc-depth=2

echo "Gotovo: report/izvestaj.docx"
