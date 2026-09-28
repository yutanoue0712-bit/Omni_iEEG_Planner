# Omni-iEEG Planner — 1.0-RC20

Omni-iEEG Planner is a local Python desktop prototype for viewing
patient-specific MRI, CT, FreeSurfer outputs, intracranial electrodes and
analysis results. This preparation copy identifies the existing RC20 code.

**Research and education only. Do not use this prototype for clinical
decision-making.** Clinical accuracy has not been established.

Repository: [yutanoue0712-bit/Omni_iEEG_Planner](https://github.com/yutanoue0712-bit/Omni_iEEG_Planner)

## License and release status

The original application code is licensed under the [MIT License](LICENSE).
Copyright (c) 2026 Yuta Tanoue, Masaki Izumi, and Hiroki Nariai.

This RC20 source snapshot is prepared for publication. The actual release
date and DOI have not yet been assigned.

Third-party portions retain their own terms. See [NOTICE](NOTICE),
[third-party licenses](THIRD_PARTY_LICENSES.md) and
[the RC20 provenance review](docs/LICENSE_AUDIT_RC20.md).

## Installation on Windows

Use Python 3.12. Open PowerShell in this repository folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m brain_viewer --demo
```

The demo generates synthetic anatomy and electrodes. Patient images, saved
cases and a standard-brain template do not need to be downloaded for it.
Installing dependencies requires network access. A desktop/OpenGL-capable
environment is needed for the GUI.

After setup, `Start_Viewer.cmd` starts the normal workspace. To open the
synthetic demo again, use the last command above.

Optional ANTsPy postoperative correction uses a separate environment.
After installing the main environment, run `Setup_Postop_ANTs.cmd`.
The separate versions are recorded in `requirements-postop-ants-lock.txt`.

Windows is the current development platform. A new-machine installation
and cross-platform verification are separate from this preparation step.

Local verification on the current Windows PC passed 201 automated tests,
9 synthetic GUI workflows and the normal launcher check. See the
[RC20 verification record](docs/VALIDATION_RC20.md) for the tested scope
and the clean-install startup correction.

## Main functions

- Patient-space MRI/CT display and image registration.
- FreeSurfer surface and label display using user-supplied outputs.
- CT-based electrode candidates with manual review and editing.
- Editable regions, diffusion tools and postoperative image comparison.
- Excel analysis-result display and PNG/MP4 export.

Read the [getting-started notes](docs/USAGE.md) and
[analysis-result instructions](docs/ANALYSIS_RESULTS.md).

## Citation

If you use Omni-iEEG Planner in research, publications, or presentations,
please cite the software and specify the version you used.

Until a DOI or associated publication is listed here, use this software
reference for RC20:

Tanoue, Y., Izumi, M., & Nariai, H. *Omni-iEEG Planner*
(version 1.0-RC20) [Computer software].
[GitHub repository](https://github.com/yutanoue0712-bit/Omni_iEEG_Planner).

Citation details are also provided in [CITATION.cff](CITATION.cff).
A Zenodo DOI and references to a related preprint or paper will be added
when available. Once listed, please cite the software DOI for the version
you used and the associated publication.

Citation is requested for scholarly credit; it does not add conditions to
the software license.

## Included files

This source copy contains Python source, synthetic test code, translations,
dependency lists and documentation. It does not include patient images,
saved cases, Excel results, screenshots, reference MATLAB source, Python
environments, Qt binaries or FFmpeg binaries. Dependencies installed by the
user have their own license conditions.

## 日本語での開始方法

本ソフトウェアは研究・教育目的の試作です。臨床判断には使用しないでください。
まず上の3行を実行すると、患者データを使わない合成デモを開けます。
版の名前は `1.0-RC20` として残します。

独自コードのライセンスは [MIT](LICENSE) です。著作権者の名義は
Yuta Tanoue、Masaki Izumi、Hiroki Nariai の3名として確定しました。
実際の公開日と DOI は公開時に記入します。
公開までの作業状況は[準備手順](docs/GITHUB_ZENODO_RC20.md)を参照してください。

## 引用のお願い

本ソフトウェアを研究・論文・学会発表に使用した場合は、
**Omni-iEEG Planner と使用したバージョンを引用してください。**
引用例は上記の「Citation」に、著者・バージョンなどの引用情報は
[CITATION.cff](CITATION.cff) に記載しています。

Zenodo の DOI と関連するプレプリント・論文の引用情報は、公開後に追記します。
DOI の発行前は、ソフトウェア名・著者名・使用バージョン・GitHub の公開先を
引用に含めてください。

引用は学術上のお願いであり、ソフトウェアの利用条件を追加するものではありません。
