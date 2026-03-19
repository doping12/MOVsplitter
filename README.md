# split-mov

## プロジェクト概要
Now Loadingの画面と曲タイトル表示を検知してNow Loadingを削除しつつ分割します。

デレステのMVリストを録画した時に手動で分割するのが面倒だったので作りました。
完全趣味で作っているので基本的に保守しません。
Codexに全部書かせたので処理ほとんど把握してないです。

多分ですけど、loadingの中のloadingとtiltleのサンプル動画を変えたら似たようなシチュエーションで使えるかもしれませんね。

## 動作確認環境
- Python 3.11+
- 依存ライブラリ: `opencv-python`, `numpy`, `scikit-image`, `PyYAML`, `matplotlib`
- 検証OS: macOS 15.7.3 のみ

## インストール
```bash
git clone MOVsplitter
cd MOVsplitter

# uv
uv venv
source .venv/bin/activate
uv pip install -e .

```

## 使い方

### 1. 全て自動実行 （解析 + 分割）
```bash
python -m split_mov \
  "/Users/ykmbp/Workspace/git/MOVsplitter/testdata/movie.mp4"
```

### 2. visualize (解析) のみ実行
```bash
python -m split_mov \
  "/Users/ykmbp/Workspace/git/MOVsplitter/testdata/movie.mp4" \
  --visualize-only
```

### 3. visualize の結果を使って分割を実行
```bash
# まず visualize-only で check_frames と cut_table.csv を作る
python -m split_mov \
  "/Users/ykmbp/Workspace/git/MOVsplitter/testdata/movie.mp4" \
  --visualize-only

# 必要なら cut_table.csv を編集

# cut_table.csv に従って切り出し
python -m split_mov \
  "/Users/ykmbp/Workspace/git/MOVsplitter/testdata/movie.mp4" \
  --cut-table \
  "/Users/ykmbp/Workspace/git/MOVsplitter/split_mov_YYYYMMDD_HHMMSS/cut_table.csv"
```

cut_table.csvを使えば調整できます。

## 出力
`--output-dir`未指定時は、カレント配下に`split_mov_YYYYMMDD_HHMMSS/`を作成して出力します。

通常実行時に自動出力される主なファイル:
- 分割点を示すスコア関係
  - `check_timeline.png` (全体のスコア変動と検知されたloading画面)
  - `check_timeline.html` (同上)
  - `score_timeline.txt` (スコア変動データ)
  - `check_boundary_plots/boundary_*_combined.png` (検知された境界前後を拡大したもの)
- 分割点のフレーム
  - `check_frames/*.jpg`
- `cut_table.csv` (分割リスト)
- `*_1.mp4` (分割された動画は連番で出力)

## Tips
### 微調整
`head_fine_tune_frame_offset` / `tail_fine_tune_frame_offset` は、最終境界をフレーム単位で微調整する設定です。

- 値を `+` にする: 境界を後ろへ送る（開始が遅くなる / 終了が遅くなる）
- 値を `-` にする: 境界を前へ寄せる（開始が早くなる / 終了が早くなる）

目安:
- loading が残る場合: `tail_fine_tune_frame_offset` を `-1`, `-2` 方向で調整
- 本編を削りすぎる場合: `head_fine_tune_frame_offset` を `+1`, `+2` 方向で調整
  - examleのYAMLに書いてある通り、+-1とかだと全然気が付かないので10くらいで一旦調整することをお勧めします。

### 設定
config用のYAMLを書き換えて調整します。
---

## YAML設定項目
- `coarse_sample_fps`: 全体の粗検知で使うFPS
- `coarse_boundary_sample_fps`: 粗境界再探索で使うFPS
- `refine_sample_fps`: 高精度境界探索で使うFPS
- `refine_window_sec`: 境界の前後で再探索する秒数（±）
- `brightness_change_threshold`: 輝度変化のしきい値
- `histogram_diff_threshold`: ヒストグラム差分のしきい値
- `ssim_threshold`: SSIMのしきい値
- `low_information_threshold`: 低情報量判定のしきい値
- `color_std_threshold`: 色分散判定のしきい値
- `frame_diff_threshold`: フレーム差分のしきい値
- `freeze_min_duration_sec`: 静止状態とみなす最小継続秒数
- `loading_min_duration_sec`: loading区間として採用する最小継続秒数
- `merge_gap_sec`: 近接loading区間を結合する間隔秒数
- `min_segment_sec`: 出力本編区間の最小秒数
- `loading_score_threshold`: loading候補とみなす最小スコア
- `smoothing_window_sec`: 判定平滑化の窓秒数
- `dark_brightness_threshold`: 暗画面判定のしきい値
- `bright_brightness_threshold`: 明画面判定のしきい値
- `use_template_matching`: テンプレート照合を使うか
- `loading_template_path`: loadingテンプレート動画のパス
- `title_template_path`: タイトル画面テンプレート動画のパス
- `template_ignore_edge_sec`: テンプレート先頭/末尾の除外秒数
- `template_similarity_threshold`: テンプレート類似度の通常しきい値
- `template_strict_similarity_threshold`: テンプレート類似度の厳格しきい値
- `title_template_similarity_threshold`: タイトルテンプレ類似度の通常しきい値
- `title_template_strict_similarity_threshold`: タイトルテンプレ類似度の厳格しきい値
- `center_white_ratio_threshold`: 中央白画素比率のしきい値
- `center_edge_ratio_threshold`: 中央エッジ比率のしきい値
- `center_outer_contrast_threshold`: 中央と外側のコントラスト差しきい値
- `title_dark_brightness_threshold`: タイトル判定時の明るさ上限
- `head_edge_refine_enabled`: loading先頭側の端補正判定を有効化するか
- `head_loading_window_sec`: loading先頭側判定の探索秒数
- `head_min_stable_sec`: loading先頭側で採用する最小継続秒数
- `head_density_threshold`: loading先頭側で採用する最小密度
- `head_template_similarity_threshold`: loading先頭側のテンプレ類似しきい値
- `head_jump_min_score`: loading先頭側のjump判定最小スコア
- `head_jump_delta`: loading先頭側のjump判定最小変化量
- `head_fine_tune_frame_offset`: loading先頭側の最終境界フレーム補正
- `tail_edge_refine_enabled`: loading末尾側の端補正判定を有効化するか
- `tail_loading_window_sec`: loading末尾側判定の探索秒数
- `tail_min_stable_sec`: loading末尾側で採用する最小継続秒数
- `tail_density_threshold`: loading末尾側で採用する最小密度
- `tail_template_similarity_threshold`: loading末尾側のテンプレ類似しきい値
- `tail_jump_min_score`: loading末尾側のjump判定最小スコア
- `tail_jump_delta`: loading末尾側のjump判定最小変化量
- `tail_fine_tune_frame_offset`: loading末尾側の最終境界フレーム補正
- `copy_cut_tolerance_sec`: `-c copy`切り出しの許容誤差秒数

## CLIオプション
- `input_file`（位置引数）: 入力動画ファイル（`.mp4` / `.mov`）
- `--output-dir`: 出力ディレクトリ
- `--build-cut-table-from-dir`: `check_frames`から`cut_table.csv`を作成
- `--cut-table`: 指定CSVの区間で切り出し実行
- `--cut-table-out`: `cut_table.csv`の出力先
- `--parallel`: 切り出し並列数
- `--config`: 設定YAML/JSONのパス
- `--min-segment-sec`: `min_segment_sec`をCLIから一時上書き
- `--dry-run`: 実際には切り出さず計画だけ表示
- `--verbose`: 詳細ログを表示
- `--export-report`: JSONレポート出力先
- `--export-preview`: プレビューCSV出力先
- `--check-boundary-window-sec`: 境界plotの前後表示秒数（±）
- `--check-jpeg-quality`: `check_frames`のJPEG品質
- `--check-frame-width`: `check_frames`の最大幅px
- `--visualize-only`: 解析と可視化のみ実行
- `--keep-temp`: 現状は予約オプション
- `--output-ext`: 出力拡張子を`mp4/mov`で明示
