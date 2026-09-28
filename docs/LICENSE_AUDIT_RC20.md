# RC20 ライセンス・コード由来の確認記録

確認日：2026-09-28。対象：Omni-iEEG Planner Prototype 1.0 RC20。

**最新の確認事項：** 公開許可に加え、ライセンスの著作権者名義もユーザーから確認・承認を得ました。独自コードは MIT とし、著作権者を Yuta Tanoue、Masaki Izumi、Hiroki Nariai の3名として、ルートの [LICENSE](../LICENSE) と CITATION の `license: MIT` を確定しました。初回の引用著者も同じ3名・同じ順番です。引用著者と著作権者は別の情報ですが、今回はそれぞれについて承認を得ています。実際の公開日と DOI は公開・発行時に記入します。

## 結論

**現在確認できる実装では、アプリ全体を GPL にする必要があると断定できる証拠は見つかっていません。ユーザーによる名義確認・承認を受け、ソース公開における独自部分のライセンスを MIT として確定しました。** ただし「第三者由来のものが一切ない」「コピー・翻案が絶対にない」という証明ではありません。第三者由来部分の条件と、この技術調査の限界は引き続き以下のとおりです。

以前の「参考にはしたがコピーや読み込みはしていない」という説明は、現在の通常起動経路については概ね整合しますが、作業フォルダ全体については正確ではありません。

- 参照用の Brainstorm MATLAB 原本が4ファイル、非公開の `private_reports/brainstorm_reference/` にあります。アプリに同梱する必要はありません。
- 旧試作の Brainstorm データ読込コードが `brain_viewer/brainstorm_import.py` に残り、テストからは利用されています。通常アプリからの呼出しは見つかりませんでした。
- FreeSurfer のラベル番号・名称がコードに含まれます。単に結果ファイルを読むだけの構成ではありません。この出典とライセンス文を別途記録しました。
- 動画出力の FFmpeg 実行ファイルは GPL-3.0-or-later です。BSD の Python ラッパーと混同できません。

## 調べた範囲と方法

1. `brain_viewer/` の Python 77ファイルと `tests/` の33ファイル、合計110ファイル・17,741行（今回の表示コメント追記前）を構文解析し、import と由来を示す文字列を検索しました。構文解析エラーは0件でした。
2. README、操作説明、開発タスクに記録された参照先・旧実装の説明を確認しました。
3. 特に `analysis_results.py`、`result_display.py`、`result_surface.py`、`result_rendering.py`、`surface_view.py` の関連処理、`brainstorm_import.py`、`registration.py` の旧処理、`anatomy.py`、`segmentation.py` を調べました。
4. ローカル保存の `figure_topo.m`、`figure_3d.m`、`panel_ieeg.m`、`bst_colormaps.m` と、今回公式から取得した `bst_shepards.m` を比較対象にしました。表示処理と補間式は内容を比較しました。
5. 補助的に、比較対象5ファイルとアプリ77ファイルの連続20単語・識別子の完全一致を検索しました。一致は0件です。この結果は、変数名の変更、短い断片、構造の翻案、他の出典からの転用を検出・否定するものではありません。
6. `.venv` と `.venv-ants` の実際のインストール情報を読み、別の監査専用環境に導入した `pip-licenses 5.5.5` から両環境を指定して一覧を生成しました。アプリの環境には監査ツールを追加していません。
7. FFmpeg 本体の `-L` 表示とビルド設定を確認しました。

この作業フォルダには `.git` がありません。過去の全編集履歴・全生成時の入力や、インターネット上の全コードとの比較はできていません。`gc_analysis/`、CCEP 関連、学習資料、論文原稿などの別作業は、今回の RC20 アプリ公開候補の監査範囲に含めません。患者画像の内容をこの監査のために外部へ送信していません。

## 1. Qt と依存パッケージ

| 確認項目 | 結果 |
| --- | --- |
| Qt バインディング | **PySide6-Essentials 6.11.2 / shiboken6 6.11.2** |
| PyQt5 / PyQt6 | アプリの import、固定依存一覧、調査した2環境に該当なし |
| Qt の使用モジュール | 主に QtCore、QtGui、QtWidgets。操作検証で QtTest |
| Slicer / MNE / nilearn / PyVista | アプリの import と2環境の依存一覧に該当なし。説明資料の参照先とは区別する |
| 主な直接依存 | NumPy、SciPy、NiBabel、Matplotlib、PySide6-Essentials、VTK、SimpleITK、openpyxl、imageio-ffmpeg、DIPY、pydicom、dcm2niix |
| 任意の術後補正 | 別環境の antspyx 0.6.2 を別プロセスで使用 |
| 環境全体の一覧 | 本体38パッケージ、術後補正33パッケージ。pip 自体は除外。間接依存も含む |

詳細は [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md)。PySide6 の `LGPL ... OR GPL ...` は選択肢を表すため、文字列に GPL が含まれるだけでアプリ全体が GPL になるという意味ではありません。LGPL の条件への対応は必要です。[Qt 公式](https://doc.qt.io/qtforpython-6/)、[Qt ライセンス](https://doc.qt.io/qt-6/licensing.html)。

## 2. コピー・移植候補の個別確認

| 箇所 | 実際の内容 | 今回の判断 |
| --- | --- | --- |
| 参照用 MATLAB 4ファイル | Brainstorm の著作権・GPLv3 表示を含む原本 | **第三者ソースの保存コピーが実在**。公開候補に入れない。存在だけで別の独立アプリが GPL になるわけではない |
| RC13 の色・電極表示 | Matplotlib の色表、Qt の操作部品、VTK の球・描画機能を利用。Brainstorm の MATLAB の図・グローバル状態・疎行列による接点形状対応とは構成が異なる | 操作・表示の参考という既存説明と整合。比較した処理で、直接コピーや逐語移植を示す証拠は見つからない |
| RC17 の脳表ヒートマップ | NumPy / SciPy の正規化逆二乗距離補間。15 mm、近傍4点という設定の参照履歴あり | 下記の数式・境界処理の相違を確認。設定や一般的手法の共通性だけで移植とは断定できない。ただし独立作成の完全な証明でもない |
| `brainstorm_import.py` | SciPy で Brainstorm 形式の `.mat` を読み、座標・チャンネルを解釈する旧コード | 「読込コードがない」は誤り。通常起動からの import は見つからず、`tests/test_multimodal.py` からの利用がある。データ形式を読むこと自体は Brainstorm 本体の import / ソース複製と同義ではない |
| `registration.py` の `match_existing_ct()` | 旧 Brainstorm CT の +1024 値オフセットを扱う SimpleITK 処理 | 関数は残存。アプリ・テスト内の呼出しは今回の検索で該当なし。旧実装を除外するならテスト依存も含めて別変更として扱う |
| Slicer / MNE | 設計資料で座標系・操作の参考として引用 | パッケージ import、由来を明記したソース転用の候補は今回見つからない。全上流ソースとの網羅照合は未実施 |
| MMVT / IntrAnat / CAT12 | アプリ・テスト内の関連名・import は検索該当なし | 転用の積極的な証拠は見つからない。名称を消した転用まで否定する検査ではない |
| FreeSurfer | 視床核29名称と番号、脳室・脳梁・視床プリセットのラベル番号 | **出典の明確なラベル表の再利用あり**。アルゴリズムの移植とは区別し、NOTICE と元ライセンスを追加 |
| DIPY / SimpleITK / ANTsPy | 公開 API を呼び出して処理を実行 | ライブラリ利用を確認。API 呼出しとライブラリ実装のコピーは別物。利用部品のライセンス表示は必要に応じて保持 |

### ヒートマップで確認した具体的な相違

対象は `brain_viewer/result_surface.py:13–32` と公式 [bst_shepards.m](https://github.com/brainstorm-tools/brainstorm3/blob/master/toolbox/math/bst_shepards.m) です。

| 比較点 | Omni-iEEG Planner | 比較した Brainstorm 関数 |
| --- | --- | --- |
| 近傍検索 | SciPy の cKDTree | bst_nearest |
| 重み | 各距離の逆二乗を正規化 | 最遠近傍までの距離も使う修正 Shepard 重み |
| 最遠の点 | 条件内なら4点目も寄与 | 複数近傍の場合、最後の点を重み行列から外す |
| 半径 | 各近傍点について半径内か判定 | 最も近い点の距離で頂点全体を除外 |
| 欠損 | 有限値のみでその時点の重みを再正規化 | 比較した関数は値を受け取らず、幾何から重み行列を作る |
| ゼロ距離 | 一致点を優先し、複数の一致点があればその値を平均 | eps 置換と、重み合計ゼロ時の先頭点採用 |
| 結果 | 表示値の配列を計算 | 疎な補間行列を返す |

「式が違うから著作権上絶対に問題がない」とは判断しません。ここで示すのは、単純な MATLAB → Python の行単位変換という説明を支持しない、実装上の証拠です。ソースを参照した履歴は [ANALYSIS_RESULTS.md](ANALYSIS_RESULTS.md) に残します。

## 3. 同梱データと FreeSurfer

アプリ本体とテストの公開候補には、MNI / ICBM152 の画像やアトラス本体は見つかりませんでした。アプリ本体で Python 以外のファイルは翻訳用 `translations.json` です。標準脳の自動取得コードも今回の検索では見つかりませんでした。これはローカルの患者データや Python 依存パッケージの内部データすべてについて「存在しない」と主張するものではありません。

FreeSurfer については、`anatomy.py` の29個の視床核名称を公式 FreeSurferColorLUT と照合し、対応する左右58ラベルを確認しました。`segmentation.py` のプリセットも LUT の番号に依存します。画像・統計・表面ファイルを利用者が入力する部分と、コード内に置かれた表を分けて記録します。

「FreeSurfer のソースのコピーは一律不可」という説明は正確ではありません。MGH の対象ソフトウェア・データには条件付きの複製・改変・配布許諾があります。個別の第三者部分は別条件の場合があります。今回は表の抜粋・再構成を明記し、保守的に [NOTICE](../NOTICE) と [FreeSurfer ライセンス全文](../LICENSES/FreeSurfer-LICENSE.txt) を添えました。[公式ライセンス](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense)。

## 4. FFmpeg を見落とさない

`imageio-ffmpeg 0.6.0` の Python ラッパーは BSD-2-Clause ですが、この Windows 環境の FFmpeg は **7.1 / GPL-3.0-or-later** です。`--enable-gpl --enable-version3` と libx264 の利用を確認しました。

`brain_viewer/result_export.py` は FFmpeg を別プロセスとして起動し、標準入力へ RGB 動画フレームを渡しています。今回の初回公開候補はアプリのソースと依存関係指定のみとし、`.venv`、FFmpeg、Qt 等の実行ファイルはコピーして配布しません。依存関係のインストール時には、利用者が入手するパッケージの条件が適用されます。

別プロセスでの利用だけを根拠に、常にライセンス上独立だと断言はしません。一方、FFmpeg が環境内にあるだけで全アプリを直ちに GPL とする説明も不正確です。EXE、インストーラー、実行環境付き ZIP を将来配布する際は、対応ソース・ビルド情報・ライセンス表示を改めて確認します。[FFmpeg 公式](https://ffmpeg.org/legal.html)、[GNU FAQ](https://www.gnu.org/licenses/gpl-faq.html#MereAggregation)。

## 5. 提示された説明の修正点

- MIT / BSD は「無条件」ではありません。配布内容に応じた著作権表示、許諾文、免責条項などの保持が必要です。SimpleITK / ANTsPy は Apache 系であり、すべてを MIT / BSD とまとめません。
- GPL の名前が1回出る、一行似ている、PC に GPL ソフトが入っている、というだけの判定はできません。保護される表現のコピー・翻案か、組み合わせ方、配布対象、ライセンスの版や例外を確認します。
- 一般的な手法・アイデアと、その具体的なコード表現は別です。論文の手法を実装しても、文章・図の転載や特許等まで無条件になるという意味ではありません。[米国 Copyright Office](https://www.copyright.gov/help/faq/faq-protect.html)。
- LICENSE という名前のファイルを作るだけでは足りず、オープンソースとしての利用・改変・再配布を許す内容と、適用する権利が必要です。[GitHub の説明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)。
- 研究用途の注意は README に書きます。MIT 本文へ「研究に限る」「商用禁止」などの追加制限を入れると、標準 MIT やオープンソースの説明と食い違います。[Open Source Definition](https://opensource.org/osd)。

## 成果物と公開時に記入する項目

- 第三者情報：`THIRD_PARTY_LICENSES.md`、`NOTICE`、`LICENSES/FreeSurfer-LICENSE.txt`。
- 自動取得の原表：`docs/licenses/pip-licenses-main.md`、`docs/licenses/pip-licenses-ants.md`。
- 引用情報：`CITATION.cff`。確認済みの著者3名・ORCID・版・リポジトリ URL と `license: MIT` を記載。実際の公開日と未発行の DOI は、公開・発行時に記入します。
- 正式な MIT 本文：[LICENSE](../LICENSE)。著作権者は Yuta Tanoue、Masaki Izumi、Hiroki Nariai の3名。旧下書きは公開対象から外し、ローカルの作業記録に保管しました。
- 研究・教育用の注意と合成デモの起動手順を README に追加。

最初の由来調査では、依存一覧と両環境の固定版一覧、CITATION の YAML 構文、ローカル文書リンク、FreeSurfer ライセンスと取得原文のバイト一致を確認し、Python 110ファイルの構文解析を行いました。この時点のアプリ側の変更は出典コメントのみでした。その後、分離した公開用コピーで初回起動を修正し、自動テスト201件と画面操作9種類に合格しています。[動作確認記録](VALIDATION_RC20.md)に、その検証範囲と制限を記録しています。今回のライセンス確定ではアプリ本体や依存バージョンを変更していません。

ユーザーが [GitHub リポジトリ](https://github.com/yutanoue0712-bit/Omni_iEEG_Planner) を作成し、URL は引用情報に反映済みです。MIT と3名の権利者表示を確定し、公開用コピーの動作確認も完了しています。こちらからのファイル送信・Public 化・tag・Release・Zenodo 登録はまだ実施していません。公開時には実行環境や私的な作業記録を除いたソースだけを登録し、実際の公開日を記入します。[初めての公開手順](GITHUB_ZENODO_RC20.md)。

この記録は技術的な確認結果と実務上の候補の整理であり、非侵害や全利用条件への適合を保証する法律意見ではありません。
