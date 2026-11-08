# Fill in the table of contents of the generated report and export PDF copies (needs Microsoft Office).
#   python -m src.make_report
#   powershell -ExecutionPolicy Bypass -File tools\finalize_report.ps1
# python-docx cannot compute page numbers, so the generated .docx has an empty contents field until Word updates it.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$docx = Join-Path $root "reports\Muhafiz_sol_Final_Report.docx"
$pptx = Join-Path $root "reports\Muhafiz_sol_Slides.pptx"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$tmp = Join-Path $env:TEMP "muhafiz_report_toc.docx"
try {
    # open read-only (an editable open of a freshly generated file can stall on a hidden dialog), save a copy
    $doc = $word.Documents.Open($docx, $false, $true, $false)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.SaveAs2($tmp, 16)
    $doc.ExportAsFixedFormat((Join-Path $root "reports\Muhafiz_sol_Final_Report.pdf"), 17)
    Write-Output ("report: {0} pages, contents updated" -f $doc.ComputeStatistics(2))
    $doc.Close($false)
} finally { $word.Quit() }
Move-Item -Force $tmp $docx

$ppt = New-Object -ComObject PowerPoint.Application
try {
    $pres = $ppt.Presentations.Open($pptx, $true, $false, $false)
    $pres.SaveAs((Join-Path $root "reports\Muhafiz_sol_Slides.pdf"), 32)
    Write-Output ("slides: {0} slides exported to PDF" -f $pres.Slides.Count)
    $pres.Close()
} finally { $ppt.Quit() }
