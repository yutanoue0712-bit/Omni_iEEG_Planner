# RC20 を初めて GitHub・Zenodo に公開する手順

RC20 の公開準備メモ。ユーザーが GitHub アカウントとリポジトリを作成しました。

## 現在決まっていること

- 公開許可：取得済み。
- アプリ名：Omni-iEEG Planner。
- 公開する版：`1.0-RC20`。この版の名前を残す。
- 初回の引用著者：Yuta Tanoue、Masaki Izumi、Hiroki Nariai。この順番。
- 独自コードのライセンス：MIT。著作権者は同じ3名として、ユーザーの確認・承認済み。[LICENSE](../LICENSE) と CITATION に反映済み。
- 公開用コピーの動作確認：自動テスト201件、画面操作9種類、通常起動が合格。ユーザー本人も起動を確認済み。[確認記録](VALIDATION_RC20.md)。
- 公開先 URL：[yutanoue0712-bit/Omni_iEEG_Planner](https://github.com/yutanoue0712-bit/Omni_iEEG_Planner)。CITATION に反映済み。
- 実際の公開日と DOI は、公開・発行時に記入する。一般向けの取得ではリポジトリの内容・公開設定は確認できていません。

`CITATION.cff` は「このアプリを論文などで引用するときの書誌情報」です。`authors` は引用に載せる著者で、LICENSE の著作権者欄とは別です。GitHub ではファイルを編集して著者の追加・訂正ができます。Zenodo 公開後もレコードの著者情報は編集でき、DOI は維持されます。GitHub の変更が既に保存された版のファイルへ自動反映されるわけではないため、Zenodo の表示情報も別途更新します。[GitHub の引用ファイル説明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)、[Zenodo の公開済み情報の編集](https://help.zenodo.org/docs/deposit/manage-records/#edit-published-records)。

## まず覚える4つの言葉

| 名前 | 意味 |
| --- | --- |
| GitHub リポジトリ | アプリのコードと説明書を置く、履歴付きのフォルダ |
| LICENSE | 他の人に、何を許可するかを示す文書。公開だけでは利用・改変・再配布を十分に許諾したことにならない |
| tag / Release | `v1.0-RC20` という名前で、どの時点のコードかを固定し、説明付きで配布するもの |
| Zenodo / DOI | 公開した版を保存し、論文で引用するための恒久的な識別子を付ける仕組み |

今回の最初の公開は **Python のソースコード** を対象にするのが扱いやすい方法です。利用者は Python と必要なライブラリを手順に沿って導入します。ダブルクリックだけで動く EXE 配布は別作業になります。

## 提示された3行は何をするのか

これらはアプリの画面や Python の `>>>` に入力するコードではなく、ターミナルで実行する指示です。元の引用の `lua` は表示用の分類で、この3行が Lua プログラムという意味ではありません。

### 1. 調査用の道具を入れる

```text
pip install pip-licenses
```

`pip` は Python の追加部品を導入する道具、`install` は導入、`pip-licenses` は導入済み部品のライセンス情報を一覧にする道具です。この行だけではアプリにライセンスは付かず、GitHub に何かを送信することもありません。通常は道具の取得のためインターネットに接続します。

Python 環境が複数あると、単に `pip` と書くと別の環境を調べることがあります。今回は RC20 の環境を変えず、`private_reports/license_audit_rc20/tool-env/` という別の監査環境に pip-licenses を導入しました。

### 2. ライセンス一覧をファイルに保存する

```text
pip-licenses --format=markdown --with-urls > THIRD_PARTY_LICENSES.md
```

- `pip-licenses`：選択された Python 環境の導入済みパッケージを調べる。
- `--format=markdown`：GitHub で表として表示しやすい形式にする。
- `--with-urls`：各パッケージのホームページ等も列に入れる。
- `>`：画面に出す代わりにファイルへ保存する。**同じ名前の既存ファイルは上書きされる。**
- `THIRD_PARTY_LICENSES.md`：保存する文書の名前。

`pip freeze` も、この環境に導入されている部品と版の一覧です。アプリが import する部品だけの一覧ではありません。間接的に必要な部品や未使用の部品も含まれ得ます。

pip-licenses は、主にパッケージが申告した情報を読む道具です。他プロジェクトから貼り付けたコード、MATLAB の移植、データの利用条件、ラッパーに含まれる FFmpeg 本体などを全部自動で判定するものではありません。「表に GPL がないから MIT でよい」という証明にもなりません。

今回の原表は [本体38件](licenses/pip-licenses-main.md) と [術後補正33件](licenses/pip-licenses-ants.md) です。読みやすくした一覧と手動確認事項は [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) にあります。Supplementary Table S1 の土台として使えますが、環境名・直接/間接依存・用途を残してください。

同じ一覧をこのPCで再生成する場合は、PowerShell でプロジェクトフォルダを開き、次を実行します。手動で追記した `THIRD_PARTY_LICENSES.md` は上書きしません。

```powershell
.\private_reports\license_audit_rc20\tool-env\Scripts\python.exe -m piplicenses --python .\.venv\Scripts\python.exe --format=markdown --with-urls --output-file private_reports/license_audit_rc20/pip-licenses-main.md
.\private_reports\license_audit_rc20\tool-env\Scripts\python.exe -m piplicenses --python .\.venv-ants\Scripts\python.exe --format=markdown --with-urls --output-file private_reports/license_audit_rc20/pip-licenses-ants.md
```

この監査用環境はローカルだけのものです。第三者が再生成する場合は、自分の監査用環境に pip-licenses を導入し、調べる Python の場所を指定します。[pip-licenses 公式説明](https://github.com/raimon49/pip-licenses#option-python)。

### 3. コピー・移植の手がかりを検索する

```bash
grep -rniE "copyright|adapted from|ported from|based on|brainstorm|mmvt|intranat|slicer|mne" --include="*.py" .
```

| 部分 | 意味 |
| --- | --- |
| `grep` | ファイルの文章を検索する |
| `-r` | 下のフォルダも検索する |
| `-n` | 見つかった行番号を表示する |
| `-i` | 大文字・小文字を区別しない |
| `-E` | `|` を「または」として使える検索形式にする |
| `copyright` | 著作権表示の候補 |
| `adapted from` / `ported from` / `based on` | 改変元・移植元・基にしたものの記述の候補 |
| ソフト名 | 出典、依存、説明、入力形式の候補 |
| `--include="*.py"` | Python のソースファイルだけを調べる |
| 最後の `.` | 現在のフォルダから調べる |

**見つかった行は「調べる候補」であって、コピー認定ではありません。** コメントに「Brainstorm は使用しない」と書いてあっても検索に出ます。反対に、出典表示のない転用は見つからないことがあります。Python だけの検索では参照原本の `.m` も漏れます。

Windows の PowerShell に grep は標準では付いていません。この環境では次の同等検索を利用できます。巨大な実行環境や患者フォルダを検索対象に含めません。

```powershell
rg -n -i "copyright|adapted from|ported from|based on|brainstorm|mmvt|intranat|cat12|slicer|mne" brain_viewer tests -g "*.py"
```

## 実際の公開はこの順番

### 1. ライセンスと名義を確定する（完了）

公開許可と著作権者の名義は確認済みです。独自コードのライセンスを MIT とし、
著作権者を Yuta Tanoue、Masaki Izumi、Hiroki Nariai の3名として、
正式な [LICENSE](../LICENSE) を置きました。CITATION の著者欄も同じ3名で、
ライセンス欄は `MIT` に設定済みです。

MIT は利用・改変・再配布・商用利用を許し、著作権表示と許諾文の保持を求め、
無保証とするライセンスです。改変版のソース公開は義務付けません。
[MIT 本文](https://opensource.org/license/mit)。

MIT は第三者由来部分の条件を消しません。`NOTICE`、
`LICENSES/FreeSurfer-LICENSE.txt` と `THIRD_PARTY_LICENSES.md` は一緒に保持します。

### 2. 公開用のフォルダを別に作る（完了）

分離した公開用フォルダ `Omni_iEEG_plannner_v1RC20` を準備し、動作確認を完了しました。
公開対象はアプリ本体 `brain_viewer/`、テスト `tests/`、依存関係4ファイル、起動・セットアップ2ファイル、README、LICENSE、NOTICE、LICENSES、CITATION、公開向けに確認した操作説明です。

動作確認のために追加した `.venv/`、`.venv-ants/`、`local_data/`、`patients/`、`private_reports/` は公開対象に含めません。

**現在の `7_application` を丸ごとアップロードしないでください。** 患者データ、保存結果、論文作業、参照用ソース、Python 環境が同じ場所にあります。`sample_images/` は内蔵の合成デモの置き場ではなく、手元の入力画像です。内蔵デモは `--demo` で生成します。

`.gitignore` は Git での新規登録を抑止する設定であり、ZIP 作成やブラウザからの手動アップロードを自動で除外する仕組みではありません。公開フォルダでは実ファイルの一覧を確認します。旧 Brainstorm 読込コードを整理する場合は、対応テストとの関係を保ち、RC20 の計算動作とは分けて変更を記録します。

### 3. 作成済みのGitHubリポジトリを使う

公開先は作成済みの
[yutanoue0712-bit/Omni_iEEG_Planner](https://github.com/yutanoue0712-bit/Omni_iEEG_Planner)
です。このリポジトリを使います。用意した README と LICENSE を登録します。

`CITATION.cff` の `repository-code` と `license: MIT` は反映済みです。
`date-released` は実際の公開日にします。引用例の `2026-09-29` を
未確認のまま公開日として固定しません。

### 4. コードを登録し、公開する

初回は GitHub の `Add file` → `Upload files`、または GitHub Desktop を利用できます。公開用フォルダに選んだファイルだけを登録します。Private で準備した場合は、内容を確認してから Public にします。

公開用コピーで依存関係を導入し、`--demo` と通常起動の確認を完了しました。[動作確認記録](VALIDATION_RC20.md)を参照してください。このPCでの確認であり、別PCでの動作確認は別途必要です。

### 5. Release を作る前に Zenodo をつなぐ

[Zenodo](https://zenodo.org/) にサインインし、GitHub アカウントを連携します。プロフィールメニューの GitHub 画面で `Sync now` を押し、公開リポジトリのスイッチをオンにします。GitHub の認証はご本人のアカウントで行います。[公式手順](https://help.zenodo.org/docs/github/enable-repository/)。

### 6. RC20 の Release を作る

GitHub のリポジトリで `Releases` → `Draft a new release` を開き、確認済みのコードに対して tag **`v1.0-RC20`** を作ります。Release の名前も同じ版と分かるものにし、説明に RC20 の範囲、研究用途、ソース配布であること、導入方法を記載します。RC は release candidate の意味なので、試験版であることも説明します。

Zenodo への取り込みを確認するまで、DOI を発行済みとは書きません。単に tag を付けるだけでなく、Release の公開が必要です。[Zenodo 公式手順](https://help.zenodo.org/docs/github/archive-software/github-upload/)。

### 7. 発行された DOI を論文へ記載する

Zenodo の実際のレコードを開き、版、著者、ライセンス、保存されたソースを確認します。RC20 を特定して引用するときはその版の DOI を使います。発行までの時間は内容量や混雑で変わり、「必ず5分」ではありません。

DOI が出てから README / CITATION と原稿の Availability を更新します。既存 tag の内容を後から動かして DOI を埋めるのではなく、更新の記録を残します。Zenodo 側のメタデータも確認してください。

## 今回完了したこと

ライセンスの技術確認、依存一覧、出典記録、3名を著作権者とする正式な MIT LICENSE、引用情報、README の研究用途・導入説明を整えました。分離した公開用コピーは自動テスト201件と画面操作9種類に合格し、ユーザー本人も通常起動を確認しました。GitHub の公開先 URL も CITATION に反映済みです。こちらからのファイル送信、Public 化、Release 作成、Zenodo 連携はまだ行っていません。
