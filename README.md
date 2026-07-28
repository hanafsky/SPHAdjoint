# SPHAdjoint.jl

[![CI](https://github.com/hanafsky/SPHAdjoint/actions/workflows/CI.yml/badge.svg)](https://github.com/hanafsky/SPHAdjoint/actions/workflows/CI.yml)

Julia + KernelAbstractions.jl による **微分可能 2D WCSPH**。
diffSPH (Winchenbach & Thuerey, JCP 2026) がやっていることを、
Apple Silicon / Metal で走る形に置き換えるための最初の一歩。

前進も随伴も同一の KernelAbstractions カーネルなので、`backend` を
差し替えるだけで **CPU / Metal / CUDA / ROCm** が切り替わる。

---

## セットアップ

パッケージ操作は**人間が Pkg REPL で行う**（このリポジトリのスクリプトは
`Pkg.add` を一切呼ばない）。プロジェクト直下で `julia` を起動し、`]` を押して：

```julia-repl
(@v1.12) pkg> activate .
(SPHAdjoint) pkg> instantiate          # KernelAbstractions, Atomix, Metal, GLMakie
(SPHAdjoint) pkg> add Literate         # ノートを書き出す場合
```

`Metal`（Apple GPU）と `GLMakie`（可視化 / インタラクティブ版）は `[deps]` に
入っている。`Literate` はスクリプト側だけで使うので入れていない。

動作確認：

```julia-repl
(SPHAdjoint) pkg> test
```

`test/runtests.jl` は 4 つの testset を持つ。**セットアップ直後にまずこれを通すこと。**

| testset | 何を見るか |
|---|---|
| カーネル W / F / G の整合性 | `∇W = F(q)·d` の前提と、サポート境界での滑らかさ |
| セルリスト vs 総当たり密度 | counting sort とアトミックが正しいか（下記チェックポイント 1〜3 の一発検出） |
| 随伴 vs 中心差分 | 本体。dJ/dX₀, dJ/dV₀, dJ/dθ |
| Brinkman 抗力が効いている | 設計場が物理に効いているか |

---

## 走らせる順番

```console
julia --project=. scripts/01_gradcheck.jl    # ① 勾配検証（CPU + Float64 で必ず）
julia --project=. scripts/02_dambreak.jl     # ② 前進のみ + ステップ/秒の計測
julia --project=. scripts/03_optimize.jl     # ③ トポロジー最適化
julia --project=. scripts/04_interactive.jl  # ④ マウスで固体を塗る（GLMakie）
julia --project=. scripts/05_metal_bench.jl  # ⑤ Metal の正しさとスループット
```

⑤で出る「ステップ/秒」が④の `nsub`（1 フレームあたりの物理ステップ数）を
決める根拠になる。実測は下の「性能」を参照。

### Metal に載せる

スクリプト冒頭の 3 行を差し替えるだけ。**カーネルは一行も変えない。**

```julia
using Metal
const T = Float32          # Apple GPU は Float64 を持たない
backend = MetalBackend()
```

ただし **勾配検証は必ず CPU + Float64 で**。Float32 では中心差分側の丸めが
効いて 3〜4 桁しか合わない（それ自体は正常）。

---

## julia-mcp で作業する場合

```python
julia_eval('include("scripts/01_gradcheck.jl")', env_path="/abs/path/to/SPHAdjoint")
```

- `src/` を編集したら Revise が拾うので `julia_restart` は不要
- テストは `env_path` を `test/` にすると `TestEnv.activate()` が自動で走る：
  `julia_eval('include("runtests.jl")', env_path=".../SPHAdjoint/test")`
- `scripts/` は全部 Literate.jl 形式。ノートに落とすには：
  `using Literate; Literate.markdown("scripts/01_gradcheck.jl", "notebooks/")`

---

## 設計の要点：なぜ Enzyme ではなく手書き随伴か

Julia で微分可能シミュレーションと言えば Enzyme.jl だが、**Enzyme の GPU 対応は
事実上 CUDA 中心**で、Metal 上を通すのは現状かなり険しい。ここが
「Julia × Metal で diffSPH」の一番の壁になる。

回避策として、SPH の随伴を**すべて gather に書き直した**。近傍関係は対称
（j が i の近傍 ⟺ i が j の近傍）なので、随伴に現れる `x̄_j += …` の
近傍への書き戻しは、スレッド i が自分の近傍 j を舐めながら「i が近傍側を
演じる分」も一緒に足す形に畳める。ペアごとの寄与を

```
da   = ā_i - ā_j
ad_d = (ā_j - ā_i) · d_ij
ad_v = (ā_i - ā_j) · (v_i - v_j)
```

にまとめると、全項が i と j のデータだけで閉じる。結果として：

- **浮動小数のアトミックが要らない**（設計変数場 θ への書き戻しだけが例外）
- 前進カーネルとまったく同じ書き方で随伴カーネルが書ける
- Metal でも CUDA でも同じコードが走る

密度総和の随伴も綺麗に畳める：

```
∂ρ_i/∂x_i = Σ_j m F_ij d_ij ,  ∂ρ_j/∂x_i = m F_ij d_ij
  ⇒  x̄_i += Σ_j m F_ij d_ij (ρ̄_i + ρ̄_j)      ← 完全に gather
```

### 位置づけ（過大評価しないこと）

**新規性は無い。** 数学的には「疎行列の転置は、疎パターンが対称なら同じ
パターンを持つ」というだけ。随伴 = 転置なので、対称な近傍グラフの上では
scatter が gather に化ける。GPU 分子動力学では
**full neighbor list（Newton の第 3 法則を使わない）** が標準的な実装選択で、
演算量 2 倍を払ってコアレスアクセスを取るという同じトレードオフ
（[GPU MD review, arXiv:2003.14061](https://arxiv.org/pdf/2003.14061)）。
JAX-SPH / diffSPH は full pair list への `segment_sum` なので、AD
フレームワークの転置規則が同じことを自動でやっている。

engineering としての価値はある：「そのバックエンドに使える AD ツールが無い」
という詰みを、前進と同じ書き方の手書き随伴で回避できる。

### 適用条件と限界（ここが本質）

gather 化が成立するのは **近傍グラフが対称なときだけ**。

- ✅ 単一の大域的 h、カットオフ 2h → `|x_i - x_j| < 2h` は対称。本実装はこれ。
- ❌ **可変平滑化長 h_i（適応解像度）** → 非対称。j が i の 2h_i 以内でも
  i が j の 2h_j 以内とは限らない。転置の疎パターンが変わるので壊れる。
  転置リストを明示的に構築するか、アトミックに戻す必要がある。
- ❌ 相互作用が相互的でない定式化（一部の境界粒子の扱いなど）でも壊れる。

また gather 化は**空間方向の話だけ**で、時間ループのテープ／チェックポイントは
別途必要。コストは Newton 第 3 法則版に比べペア演算が 2 倍（GPU では
メモリ律速なのでほぼ無料、CPU では素直に 2 倍）。

### カーネル関数の形

∇W を `F(q)·d_ij` の形で持つのが地味に効いている（Wendland C2, 2D）：

```
q = r/h,  A = 7/(4π h²)
W(q)     = A (1-q/2)⁴ (2q+1)
∇_i W_ij = F(q) d_ij ,  F(q) = -5A (1-q/2)³ / h²
dF/dq    = G(q) = (15A/2) (1-q/2)² / h²
```

`r` で割る操作が消えるので **r→0 の特異性が無い**。さらに W も F も
サポート境界 q=2 で滑らかに 0 に落ちるため、**近傍リストの出入りで勾配が
壊れない**（diffSPH 論文が指摘している落とし穴のひとつ）。

---

## 検証状況

随伴の定式化は**数値的に検証済み**。

| 検証 | 結果 |
|---|---|
| scatter 形式 vs PyTorch 自動微分 | dJ/dX₀ **2.6e-15**, dJ/dV₀ 1.4e-15, dJ/dθ 1.0e-15 |
| **gather 形式**（= 本カーネルの式）vs PyTorch 自動微分 | dJ/dX₀ **2.7e-15**, dJ/dV₀ 1.4e-15, dJ/dθ 9.2e-16 |
| scatter 形式 vs 中心差分 | 9 桁一致 |

`tools/verify_adjoint.py`（scatter 形式）と `tools/verify_gather.py`
（gather 形式 = カーネルと一行ずつ対応）がその検証ハーネス。
Julia コードとは独立に「式が正しいか」だけを確かめる使い捨てのもの。

```console
$ pip install numpy torch
$ python tools/verify_gather.py
```

### Julia / Metal での実行確認（2026-07-26, Julia 1.12.5, Apple M2）

構文レベルの確認しかできていなかった部分を実機で確認済み。

| 確認 | 結果 |
|---|---|
| `pkg> test`（CPU + Float64） | 30/30 パス |
| `01_gradcheck.jl`（CPU + Float64） | 随伴 vs 中心差分 相対 **1e-7〜1e-10** |
| セルリスト CPU vs Metal | counts / starts / order **完全一致** |
| 密度総和 CPU vs Metal（Float32 同士） | 相対 3.5e-07 |
| **随伴 CPU vs Metal**（θ の浮動小数アトミック込み） | gθ 相対 **1.2e-06**、gX 3.3e-07 |

**Metal 上の非決定性（解決済み）**: 以前は `_cl_fill!` のアトミック cursor により
セル内の粒子順序が実行ごとに変わり、総和順序の丸めで 1 ステップあたり ~1e-7
揺らいでいた。これがカオス的に増幅し、N=1500 を 5333 ステップ積分すると
同一設計に対する目的関数が **0.29% ばらついていた**（`03_optimize.jl` の直線探索が
受理していた改善幅 0.01〜0.1 の 10〜100 倍のノイズ床）。

counting sort のあとで各セルの区間を粒子 id の昇順に整列させることで根治した
（`_cl_sort_cells!`）。**N=2072 を 5333 ステップ × 4 回で J がビット一致する。**
コストはソート 20〜25 μs で、近傍リストの再構築（既定 4 ステップごと）に
ぶら下がるためステップ時間の **1.2〜2.8%**。

前進・`gX`・`gV` はビット再現する。**`gtheta` だけは例外**で、設計変数場への
書き戻しに残った浮動小数アトミックの加算順序が実行ごとに変わる
（実測で絶対差 7e-12、`|gtheta|max` ~1e-4 に対し相対 ~1e-7 と Float32 の丸めと
同程度）。完全な決定性が要るなら、設計格子が小さいことを利用して
`galpha` を CPU に落として集約すればよい。

## 初回のチェックポイント（すべて解決済み）

1. ~~**`Atomix.@atomic a[i] += v` の返り値**~~ → 想定どおり「新しい値」を返す。
   CPU / Metal 両方で counting sort が正しく動くことを確認済み。
2. ~~**`cumsum!` の GPU 対応**~~ → Metal でそのまま動く。CPU 往復は不要。
3. ~~**Metal での Int64**~~ → 実測して採用済み。ペアループのセル走査を Int32 に
   落とすと accel が 2.23x → 2.63x（N=60000）。Apple GPU の Int64 演算は
   エミュレーションで 4〜16 倍遅い（philipturner/metal-benchmarks）。
4. ~~**`gtheta` への浮動小数アトミック**~~ → Apple8 / Metal 4 でそのまま動く。
   CPU 随伴と 1.2e-06 で一致。
5. **`04_interactive.jl` の Makie API** — GUI はまだ未実行。ここだけ残っている。

## 性能を測るときのルール（守らないと必ず間違える）

このリポジトリでは性能の主張を **3 回続けて誤った**。過大評価（2 倍→実際 1.5 倍）、
過小評価、存在しない劣化の報告。原因はすべて同じで、**異なる条件で測った数字を
比較した**ことだった。以下を最低条件とする。

1. **同一プロセス内で交互に測る。** このマシンでは CPU の測定値がセッションを
   またぐと**最大 2 倍ぶれる**（同じコード・同じハーネスで 3243 → 4984 step/s）。
   熱状態と背景負荷のドリフトなので、旧実装と新実装を同じセッションで交互に
   回して相殺すること。変更前のコードは worktree から抽出して同居させればよい
   （`git worktree add <dir> <commit>` + 旧カーネルを別名で `include`）。
2. **必ずウォームアップする。** Metal はカーネルの初回コンパイルを含めると
   **1/10** の数字が出る。「GPU が CPU より遅い」と見えたら、まずこれを疑う。
3. **複数回測って最良値を採る。** 単発の測定は 30% 以上ぶれる。
4. **合成ベンチを実カーネルの代理にしない。** 近傍リストのレイアウト差は
   合成ベンチでは ±5% だったが、実カーネルでは Metal で 1.5〜1.6 倍あった。
5. **一度に 1 つだけ変える。** `u^4` の手展開と fastmath を同時に入れて
   「fastmath が 2.5 倍」と誤って帰属した（実際は pow 排除 2.25x × fastmath 1.14x）。
6. **測定値が理屈と矛盾したら、まず測定を疑う。** 「カーネル単体は速いのに
   `simulate!` が 6 倍遅い」という矛盾から初回コンパイルの混入に気づけた。

## 性能（Apple M2 10 GPU コア, Float32, ステップ/秒）

`scripts/05_metal_bench.jl` の実測。ダムブレイクの前進のみ。

| N | CPU (4 スレッド) | Metal | GPU/CPU |
|---:|---:|---:|---:|
| 925 | 4983 | 14633 | 2.94x |
| 3750 | 2352 | 12151 | 5.17x |
| 15000 | 1194 | 6394 | 5.35x |
| 60000 | 392 | 2132 | 5.44x |

高速化は 3 段階で入っている：

**① カーネルごとの `synchronize` の除去。** 以前は `step!` で 5 回 /
`backward!` で 1 ステップ 7 回同期しており、粒子数に関係なく 1 ステップ
≒ 1.5ms の往復レイテンシに張り付いていた（N=925 で 663 step/s）。
同一 backend のカーネルは同じキューに順序どおり積まれるので、ホストが読むまで
同期は要らない。`Array()` / ホストへの `copyto!` 自体が同期点になっている
（同期無しで読んだ値と、そのあと同期して読んだ値の差は厳密に 0）。除去で
**N=925 で 7.3 倍**。CPU backend は `synchronize(::CPU) = nothing` なので影響なし。

**② ペアループの演算チューニング**（`scripts/07_kernel_tuning.jl` で変種を
実測してから採用）。N=60000 の Metal で density 1.6 倍 / accel 2.6 倍、
ステップ全体でさらに 2 倍。CPU にも 1.3 倍効いた。内訳：

| 変更 | 効果 (accel, N=60000) | 理由 |
|---|---:|---|
| `@kernel inbounds = true` | 1.29x | デバイス側境界チェックの除去（BoundsError が出ていた = 有効だった証拠） |
| r² で早期棄却 + 除算の前計算 | 2.23x | 3×3 走査候補の 65% は近傍でないのに sqrt/除算を払っていた。`p/ρ²`, `1/ρ` は粒子ごとに前計算（`eos_kernel!`）してペアループを乗算だけに |
| セル走査を Int32 | 2.63x | Metal は 64bit 整数をエミュレーションする |
| **`u^4` を書かない**（`(u²)²` に手展開） | density **2.25x** | Julia の `Float^Int` は精度補正付き `pow_body` に落ちる。`^2`/`^3` は乗算に展開されるので無害、**`^4` 以上が罠**。カーネル内の全 `^4` を排除した |
| sqrt だけ `@fastmath`（density と adj_pass2 のみ） | +1.14x | Metal の精密 Float32 sqrt は補正付き命令列、fast 版は HW 命令 1 発。誤差は数 ulp（rel ~3e-7）で GPU 非決定性の床と同水準。CPU のネイティブ fsqrt は元々 IEEE なので CPU 結果はビット不変 |

fastmath / pow の注意（`scripts/08_fastmath.jl`、切り分けの経緯ごと記録）：
- 最初「fastmath で density 2.5 倍」に見えたのは**誤帰属**だった。実験カーネルに
  `u^4 → (u²)²` の手展開が紛れており、寄与の内訳は pow 排除 2.25x ×
  fast sqrt 1.14x。単一変更で測り直して確定させた。
- **accel には効かない**（0.98〜1.06x）。近傍ごとの gather（pterm/invrho/V の読み）が
  律速で sqrt が隠れているため。IEEE のまま残した。
- `@fastmath` をブロック全体にかけると `^` が `pow_fast` になり、**Metal の
  バックエンドコンパイラ (AGXMetalG14G) が XPC 断で落ちる**。sqrt 1 箇所に
  限定して使うこと。

ワークグループサイズ（64〜512）は ±3% で誤差範囲。KA のデフォルトのままにした。

**③ 近傍リストの実体化**（`scripts/09_neighbor_list.jl` で変種を実測してから採用）。
セル幅 2h の 3×3 走査は候補 61 個を舐めて実近傍は 21 個（π/9 ≈ 35%）で、残り 65%
が空振り。しかも物理カーネルは前進 2 本 + 随伴 2 本の 4 本あるので 4 回繰り返される。

変更前のセル走査と**同一プロセス内で交互に測った**結果（各 3 回の最良値）:

| N | CPU 旧 → 新 | Metal 旧 → 新 |
|---:|---|---|
| 925 | 3900 → 4983 (1.28x) | 9124 → 14633 (1.60x) |
| 3750 | 1912 → 2352 (1.23x) | 6933 → 12151 (1.75x) |
| 15000 | 798 → 1194 (1.50x) | 3436 → 6394 (1.86x) |
| 60000 | 249 → 392 (1.58x) | 1418 → 2132 (1.50x) |

**レイアウトが効く。** 素直な CSR（offsets + 可変長 indices）は構築が 1438 μs
(N=60000) とセルリスト 138 μs の 10 倍かかって割に合わない。行長を固定して
**転置**（粒子 i の m 番目の近傍を `indices[(m-1)*N+i]` に置く）にすると、同じ m で
全スレッドが連続アドレスを触るのでコアレスし、行長が既知なので count パス・
cumsum・offsets が丸ごと不要になる（2 パス版はこの前段だけで 419 μs 使っていた）。

レイアウトは `layout = :slot / :particle` で切り替えて実測できる:

| | CPU (particle/slot) | Metal (particle/slot) |
|---|---:|---:|
| N=15000 | 1.00x | **0.65x** |
| N=60000 | 0.94x | **0.61x** |

Metal では転置が 1.5〜1.6 倍効き、CPU はレイアウトに無関心。既定は `:auto`
（GPU=slot, CPU=particle）。

### 近傍リストの再利用と、その安全条件

物理カーネルが `r² < (2h)²` で絞るので、リストが真の近傍の**上位集合**でありさえ
すれば、毎ステップ構築した場合と**厳密に同じ相互作用集合**になる。カットオフを
`2h(1+skin)` で作れば、各粒子の変位が `skin·h` 以下の間これが保たれる。

ここから、**前進と随伴で再構築のタイミングを揃える必要が無い**ことも従う
（相互作用集合が同じなら計算する関数が同じなので離散随伴は厳密なまま）。
`test/runtests.jl` に「`interval=1` と `interval=8` で実効的な相互作用集合が完全一致
する」テストを置いてある（数値は総和順序のぶん 1 ulp ずれる）。

許される再構築間隔は `K ≤ skin·h/(dt·v_max)`。ダムブレイクでは K ≤ 7.4 なので
**既定は余裕を見て 4**。変位超過と行溢れは GPU 側でフラグに立て、`simulate!` /
`backward!` の終わりで報告する。**警告が出た設定の勾配は信用しないこと**
（実測で 1e-2 ずれた）。

```julia
st = State(backend, X0, V0, p; skin = 0.2, interval = 4, maxnb = 64, layout = :auto)
backward!(ws, tape, th, p, backend; seedX, seedV, interval = 4)   # 前進と揃える
```

残る支配項は accel の gather（間接参照）。

### 随伴込みのスループット（Apple M2, Float32, ステップ/秒）

| N | | テープ有り前進 | 逆行 |
|---:|---|---:|---:|
| 925 | CPU | ~4400 | ~2500 |
| 925 | Metal | **5081** | **3567** |
| 3750 | CPU | 1367 | 659 |
| 3750 | Metal | **4366** | **2575** |
| 15000 | CPU | 503 | 311 |
| 15000 | Metal | **2470** | **1057** |

逆行が前進の 2 倍強遅いのは相場どおり（diffSPH 論文も AD オーバーヘッドの
上限を約 5 倍としており、手書き随伴の 2.3 倍はむしろ良い側）。

**テープを一本の `(2, N, nsteps)` 配列にしたのが効いた。** 以前は毎ステップ
`copy(st.X)` で新しい配列を 2 個確保しており、Metal では確保コストがそのまま
効いてテープ有りの前進が 3.2 倍遅くなっていた（4191 → 1311 step/s）。
確保を一度きりにして KA カーネルでスライスへ書き込む形にしたところ、
オーバーヘッドは **0〜7%**（測定誤差の範囲）まで落ちた。Metal では
前進 3.5 倍・逆行 2.8 倍（N=925）。CPU は元から無償だったので変化なし。

残っているボトルネック（文献調査サブエージェントの報告と合わせた整理）:

- 小さい N（〜数百）では CPU の方が速い。GPU が勝つのは N ≳ 900 から。
- **GPU/CPU 比は Apple Silicon では構造的に伸びない。** CPU と GPU が同じ
  ユニファイドメモリ帯域（M2 で STREAM 実測 ~91 GB/s）を共有するため、
  帯域が効く領域では比は「実効帯域の比」で頭打ちになる。dGPU のような
  10 倍は原理的に出ない。**KPI は比ではなく絶対時間と達成帯域**にすること。
- 次に効くとされる手（未実装）: **近傍リストの実体化（CSR 形式）+ skin 付きの
  数ステップ使い回し（Verlet 化）**。3×3 セル走査の 61 候補 → 実近傍 21 という
  divergence をリスト構築側に隔離できる。本ケースは 2D・近傍 21 個・
  随伴で同じリストを 1 ステップに 4 回使う・メモリ余裕あり、と文献の
  「Verlet が有利になる条件」が揃っている（Winkler et al. 2018, CPC 225）。
- セル幅を 2h から h にして 5×5 走査にすると候補が 61 → 約 42（〜1.3 倍上限）。
- 長時間回して粒子が混ざったら「数十ステップに 1 回のセル順再ソート」。
  短時間では効かないことを実測済み（0.94〜0.96 倍）だが、これは初期配置が
  格子でセル整合的なため。DualSPHysics も定期再ソートとして使っている。

### `04_interactive.jl` の `nsub`

dp=0.012（N=925, dt=1.5e-4）で 30fps 実時間 1 倍には 222 ステップ/フレーム必要。
Metal の 4858 step/s なら 30fps で `nsub = 162`、**実時間の 0.73 倍**で回る。

---

## 構成

```
Project.toml
src/
  SPHAdjoint.jl   モジュール本体
  kernels.jl      Wendland C2 カーネル W, F=∇W/d, G=dF/dq
  params.jl       SPHParams、平滑壁、設計変数場の双線形補間とその随伴
  neighbors.jl    一様セルリスト（counting sort、Int32 アトミックのみ）
  forward.jl      密度総和 / 加速度 / semi-implicit Euler
  adjoint.jl      手書き離散随伴（pass1: 加速度、pass2: 密度総和、design: θ）
  driver.jl       テープ、backward!、目的関数の例
scripts/          探索コード（すべて Literate.jl 形式）
  01_gradcheck.jl
  02_dambreak.jl
  03_optimize.jl
  04_interactive.jl
  05_metal_bench.jl    Metal の正しさ（CPU と突き合わせ）とスループット
  06_metal_profile.jl  段階ごとの内訳・実効帯域・並べ替えの効果
  07_kernel_tuning.jl  ペアループ変種の実測（inbounds / 早期棄却 / Int32 / wg）
  08_fastmath.jl       fastmath と pow の寄与の切り分け
  09_neighbor_list.jl  近傍リストのレイアウト比較（2 パス CSR vs 転置・行長固定）
test/runtests.jl
tools/            Python 検証ハーネス（随伴式の独立検証用）
notebooks/        Literate.jl の出力先
```

## 物理モデル

```
密度総和      ρ_i = Σ_j m W(q_ij)              （自己項を含む）
線形状態方程式 p_i = c² (ρ_i - ρ0)
圧力加速度    a^p_i = -Σ_j m (p_i/ρ_i² + p_j/ρ_j²) F_ij d_ij
Morris 粘性   a^v_i = Σ_j m (2μ/(ρ_i ρ_j)) F_ij (v_i - v_j)
重力          g
Brinkman 抗力 -α(x_i) v_i                       ← 設計変数場
平滑壁        kw · relu(penetration)²
時間積分      semi-implicit Euler
```

Tait の状態方程式（γ=7）ではなく**線形 EOS**。硬すぎる EOS は勾配が暴れるので、
まず微分が通ることを優先した。粘性も符号スイッチのある人工粘性ではなく
Morris 型（滑らか）。

**既知の制約**: 抗力項が陽解法なので `dt < 2/α_max`。固体を硬くしたくて
`α_max` を上げるとすぐ発散する。速度更新の抗力部分だけ陰的にすれば
無条件安定になる：

```
v' = (v + dt·a_rest) / (1 + dt·α)
```

微分も割り算ひとつぶんだが、**現在の随伴は陽解法版で検証済み**なので、
変更したら `01_gradcheck.jl` と `test/runtests.jl` を通し直すこと
（`src/forward.jl` の `integrate_kernel!` と `src/adjoint.jl` の抗力項の両方）。

## トポロジー最適化のやり方

粒子は Lagrangian に動くが、**設計変数は固定 Eulerian 格子**に置く。
これが粒子法トポ最適の定石で、NBPH (Liu et al., SMO 2023) の
"voxel-based friction field" と同じ発想。FEM のトポ最適の道具が
そのまま流用できる：

- Borrvall–Petersson の凸補間 `α(ρ) = α_max·q(1-ρ)/(q+ρ)`
- 線形重みの密度フィルタ（+ その随伴）
- 体積制約の二分法射影

### 実際に回して踏んだ落とし穴（`03_optimize.jl`）

随伴が正しくても、その外側で最適化は簡単に死ぬ。実際に 2 つ踏んだ。

1. **目的関数に信号が無い設定** — `nsteps=1500`（0.3 秒）では水が目標
   (0.85, 0.10) に届いておらず、2σ 内の粒子が **0 個**、`J = -1.4e-02`。
   これは先頭 1 粒子の指数関数の裾を測っているだけで、勾配は実質ノイズ。
   届く時間まで延ばすと `J` は 3 桁増える（4000 步で -4.9e+01、101/240 粒子）。
   **最適化を回す前に「初期設計で目標に何粒子届くか」を必ず見ること。**
   スクリプトは 0 個なら警告を出すようにしてある。
2. **体積制約が等式射影になっていた** — `project_volume` が
   `mean(rd) ≥ volfrac` ではなく `mean(rd) == volfrac` を強制していたため、
   全部流体 (`rd=1`) の出発点から **ステップ幅に関係なく**毎回 25% を固体に
   変える一歩が入っていた（ステップ 0 でも全節点が 0.25 動く）。J が初手で
   3 桁悪化し、以後戻らない。制約は下限なので、満たしていれば何もしないのが正解。

加えて、固定幅の最急降下では設計が [0,1] しか動けないぶん一歩が過大になりやすい。
**改善した候補しか採用しないバックトラッキング直線探索**（候補の評価は前進のみ）を
入れて単調降下にしてある。

この 3 点を直した結果：

| | 粒子数 | ステップ | J（初期 → 40 反復後） | 時間 |
|---|---:|---:|---|---:|
| CPU (Float64, dp=0.020) | 240 | 4000 | -48.816 → **-50.032**（2.5%） | 87 秒 |
| Metal (Float32, dp=0.008) | 1500 | 5333 | -337.85 → **-363.83**（7.7%） | **122 秒** |

どちらも全反復が受理される単調降下。

### 受理閾値は要らない（当初の想定と違った）

非決定性を根治する前は、同一設計に対する J が 0.29% ばらつく一方で直線探索が
受理していた改善幅は 0.01〜0.1 しかなく、**後半の受理判定がノイズに埋もれていた**。
「受理閾値をノイズ床より大きく取る」対策を検討していたが、セル内順序の整列で
J がビット再現するようになった（上記「Metal 上の非決定性」）ため、
**閾値そのものが不要になった**。この設定でも 5333 ステップ × 4 回で J が完全一致する。

代わりに効いたのは**近傍リストの再構築間隔**だった。既定の `interval = 4` では
この設定の変位上限を破っており（実測: 1 ステップ最大変位 6.96e-4 に対し許容
`skin·h` = 2.08e-3 → K ≤ 2）、近傍が欠落して勾配がずれていた。
`|v|max = 3.28` で `c = 15` に対し `v/c = 0.22` あり、既定が想定する `c/10` より速い。
`NL_INTERVAL = 2` に絞ると警告が消え、40 反復すべてが 1 回目の評価で受理される。

**刻みは 34/40 反復で上限 0.30 に張り付いている**ので、上限を上げればもっと
速く収束する余地がある（未検証）。

## メモリ

テープは全ステップの `(X, V)` を保存する素朴な実装。`Tape(backend, N, nsteps, p)`
で `(2, N, nsteps)` の配列を X 用・V 用に**一度だけ**確保し、以降は確保しない
（使い回すときは `reset!(tape)`）。Float32 で `2·4·2·N·nsteps` バイト、
N=10⁴, nsteps=10³ で約 320 MB。これを超える規模では 2 段チェックポイント
（k ステップごとに保存し、逆行時に前進を再実行）に切り替えること。

## 次の一歩

- [x] `pkg> test` を通す
- [x] `01_gradcheck.jl` を CPU + Float64 で通す
- [x] Metal のステップ/秒を測る（`05_metal_bench.jl`）
- [x] Metal 上で随伴が動くことを確認する（θ の浮動小数アトミック込み）
- [x] テープを一本の `(2, N, nsteps)` 配列にする（Metal で前進 3.5 倍・逆行 2.8 倍）
- [x] `03_optimize.jl` を Metal で回す（N=1500, 40 反復で 283 秒）
- [x] ペアループのチューニング（inbounds / r² 早期棄却 / 除算前計算 / Int32。
      Metal で density 1.6x, accel 2.6x, 全体 2x。CPU も 1.3x）
- [x] 近傍リストの実体化（転置・行長固定）+ skin 付き再利用
      （CPU 1.2〜1.6 倍、Metal 1.5〜1.9 倍）
- [x] セル内の粒子順序を決定的にする（GPU の目的関数が 0.29% 揺らいでいた問題。
      セルリスト側でソートするのが安く、ステップ時間の 1.2〜2.8% で済んだ）
- [ ] `gtheta` の決定性（残った唯一の浮動小数アトミック。相対 ~1e-7）
- [ ] セル幅 h + 5×5 走査（候補 3 割減。ただしリスト構築側にしか効かず、
      その構築は数ステップに 1 回しか走らないので優先度は低い）
- [ ] `04_interactive.jl` を実際に動かす（Makie API は未検証）
- [ ] 抗力項を陰的にする（+ 随伴の更新と再検証）
- [ ] チェックポイント化して長時間の逆伝播へ
- [ ] `04_interactive.jl` に「感度を見る」ボタン（塗る → どこに塗るべきだったかを見る）
- [ ] particle shifting を最適化問題として解く（diffSPH の目玉）
- [ ] 3D 化
- [ ] 可変 h への拡張（= gather が壊れるところ。転置リスト構築が要る）

## 参考

- Winchenbach & Thuerey, *diffSPH: Differentiable SPH for Adjoint Optimization
  and Machine Learning*, [arXiv:2507.21684](https://arxiv.org/abs/2507.21684)
  → J. Comput. Phys. 2026
- Liu et al., *Turbulent flow topology optimization ... NURBS-based particle
  hydrodynamics*, Struct Multidisc Optim 2023
- Borrvall & Petersson, *Topology optimization of fluids in Stokes flow*, 2003
- Morris, Fox & Zhu, *Modeling low Reynolds number incompressible flows using
  SPH*, JCP 1997
