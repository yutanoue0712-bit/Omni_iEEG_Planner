# RC20 動作確認記録

確認日：2026-09-28  
対象：Omni-iEEG Planner 1.0-RC20（内部バージョン 1.0.0rc20）  
結果：**このWindows PCでの動作確認は合格しました。**

## 確認した環境

- Windows 11、64ビット（build 22631）、Python 3.12.14。
- 公開用フォルダに新しく作成した .venv と .venv-ants を使用。
- 本体用 38 パッケージは requirements-lock.txt の指定バージョンと全件一致。
- ANTsPy用 33 パッケージは requirements-postop-ants-lock.txt の指定バージョンと全件一致。
- 両環境の依存関係チェックに不整合なし。
- 元の開発フォルダのコードをPythonの検索先に含めず、公開用フォルダの本体を実行。

## 結果

自動テスト **201件すべて合格**（スキップなし）。所要時間は 111.223 秒。
この中には、実際のANTsPyワーカーを使う合成データの補正・保存確認も含みます。

画面操作テストは、通常のWindows表示方式で **9種類すべて合格**しました。

| 確認項目 | 結果 | 所要時間（秒） |
| --- | --- | ---: |
| 基本操作・MRI/CT・解析表示・保存と再読み込み・画像と動画の出力 | 合格 | 79.89 |
| DTI・線維表示と保存 | 合格 | 12.06 |
| 表示画面・領域・断面切り替え | 合格 | 14.05 |
| 線維と領域の編集 | 合格 | 13.23 |
| CT断面と電極の確認・編集 | 合格 | 6.09 |
| PET・骨CTの追加と表示 | 合格 | 12.53 |
| PET脳表表示・電極レビュー | 合格 | 16.55 |
| 術後MRI・除外領域・補正・保存・処理キャンセル | 合格 | 37.75 |
| 画像の削除・領域編集・保存と復元 | 合格 | 20.08 |

最後に Start_Viewer.cmd から通常起動し、患者未登録の空の作業画面が表示されることを確認しました。
合成デモは別途 --demo で確認しています。サンプル画像の同梱は不要です。

## 確認中に修正した点

患者やサンプル画像がない初回起動では、空の作業画面を開くように
brain_viewer/workspace_ui.py を修正しました。これにより、同梱しない sample_images を
自動で読み込もうとして警告になる問題を解消しました。

tests/test_workspace_startup.py に、初回起動・デモ・既存症例・既存サンプル・
明示した入力先の扱いを確認する5件のテストを追加しました。
既存症例の再開や、全症例を削除した後の起動も検査しています。

修正対象はこの公開用コピーです。元の開発フォルダの既存ファイルは変更していません。
RC20のバージョン表記は維持しています。

## 確認範囲と制限

- 生成した合成データと自動テスト用データで確認しました。患者の実データは使用していません。
- 別PC・別OSでの動作や、臨床的な正確性を検証した結果ではありません。
- 最初に試したQtの画面非表示方式（offscreen）は、このWindows環境では
  VTK/OpenGLの初期化に失敗しました。最終結果は通常のWindows表示方式での再検証です。
- VTKの非推奨APIに関する警告は記録されていますが、最終の自動テストはすべて合格しています。

## 再確認する方法

Windowsでセットアップした環境から、次のように実行できます。
画面操作テストではテスト用のウィンドウが開閉します。

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v
    .\.venv\Scripts\python.exe -m brain_viewer --demo --smoke-test
    .\.venv\Scripts\python.exe -m brain_viewer --demo --smoke-dti
    .\.venv\Scripts\python.exe -m brain_viewer --demo --smoke-views
    .\.venv\Scripts\python.exe -m brain_viewer --demo --smoke-editing
    .\.venv\Scripts\python.exe -m brain_viewer --demo --smoke-contacts
    .\.venv\Scripts\python.exe -m brain_viewer --demo --smoke-images
    .\.venv\Scripts\python.exe -m brain_viewer --demo --smoke-refinements
    .\.venv\Scripts\python.exe -m brain_viewer --demo --smoke-postop
    .\.venv\Scripts\python.exe -m brain_viewer --demo --smoke-layer-editing

詳細ログ・合成デモの確認画像・ファイル照合記録は、
このPCの private_reports/rc20_verification に保存しています。

## GitHubへ載せるもの

この記録、本体、テスト、依存関係の一覧、説明書などを公開対象にします。
起動準備で作った .venv、.venv-ants、local_data、patients、private_reports は
PC内で使うもので、公開対象から除外します。.gitignore にも指定済みです。
ブラウザで手動アップロードする場合も、これらのフォルダを選択しないでください。

この確認記録の日付は検証日です。リリース日やDOIの確定を意味しません。
