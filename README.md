# SPHAdjoint.jl

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
(@v1.11) pkg> activate .
(SPHAdjoint) pkg> instantiate          # 本体の依存 (KernelAbstractions, Atomix)
(SPHAdjoint) pkg> add Metal            # Apple GPU で走らせる場合
(SPHAdjoint) pkg> add GLMakie          # 可視化 / インタラクティブ版
(SPHAdjoint) pkg> add Literate         # ノートを書き出す場合
```

`Metal` / `GLMakie` / `Literate` は本体の依存ではないので、`[deps]` には
入れていない。スクリプト側だけで使う。

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
```

②で出る「ステップ/秒」が④の `nsub`（1 フレームあたりの物理ステップ数）を
決める根拠になる。実時間 1 倍を 30fps で出すには `dt=2e-4` なら
1 フレーム 167 ステップ必要で、これは M 系でもかなり厳しい。
最初は実時間より遅く回すつもりで。

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

> **未実施**: Julia 処理系そのものでの実行確認。作成環境に Julia バイナリが
> 無かったため（julialang.org がネットワーク許可リスト外）、**構文レベルの
> 確認しかできていない**。数式は上表のとおり独立に検証済みなので、
> 詰まるとすれば Julia/KA の作法まわり ↓。

## 初回のチェックポイント

1. **`Atomix.@atomic a[i] += v` の返り値**
   `src/neighbors.jl` の counting sort が「新しい値を返す」前提で書いてある。
   挙動が違ったら `Atomix.modify!` に置き換える。
   → `test/runtests.jl` の「セルリスト vs 総当たり密度」が落ちたらここ。
2. **`cumsum!` の GPU 対応**
   `src/neighbors.jl` のプレフィックス和。バックエンドが `accumulate!` を
   持っていなければ、その 3 行を CPU 往復に差し替える（セル数は小さい）。
3. **Metal での Int64**
   セル添字に `Int` を使っている。Metal では `Int32` の方が速いはず。
4. **`gtheta` への浮動小数アトミック**
   ここだけ scatter。Metal で怪しければ、設計格子が小さいことを利用して
   ᾱ を CPU に落として集約する形に逃げられる。
5. **`04_interactive.jl` の Makie API**
   GUI は未実行なのでイベント周りは要デバッグ。

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

## メモリ

テープは全ステップの `(X, V)` を保存する素朴な実装。
Float32 で `4·2·N·nsteps` バイト（N=10⁴, nsteps=10³ で約 320 MB）。
これを超える規模では 2 段チェックポイント（k ステップごとに保存し、
逆行時に前進を再実行）に切り替えること。

## 次の一歩

- [ ] `pkg> test` を通す
- [ ] `01_gradcheck.jl` を CPU + Float64 で通す
- [ ] `02_dambreak.jl` で Metal のステップ/秒を測る
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
