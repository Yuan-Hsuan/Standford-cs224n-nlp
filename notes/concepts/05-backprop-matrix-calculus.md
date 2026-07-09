# 05 · Training Neural Nets 訓練神經網路 — matrix calculus & backprop 矩陣微積分與反向傳播

> **Source 出處**: CS224n Lecture 3 — *Neural net learning: gradients by hand & backpropagation*.

---

## Contents 目錄
1. The setup: SGD needs the gradient 起點:SGD 要梯度
2. Gradients by hand — matrix calculus 手算梯度:矩陣微積分
3. Gradients of the neural net (the error signal δ) 神經網路的梯度
4. ⭐ Backpropagation 反向傳播
5. In practice: autodiff & gradient checking 實務
6. One-line summary 一句話總結

---

## 1. The setup: SGD needs the gradient 起點:SGD 要梯度

A net is stacked linear layers + non-linearities: $z=Wx+b,\ a=f(z)$ (activation applied **element-wise**).
Training = **SGD**: 訓練就是梯度下降:
$$\theta^{new}=\theta^{old}-\alpha\,\nabla_\theta J(\theta)$$
where $\theta$ is **everything learnable** — weights, biases, *and even the word vectors*. 連詞向量也一起學。
So the whole game is: **how do we compute $\nabla_\theta J$?** Two ways — (1) **by hand** (§2–3), (2)
**algorithmically = backprop** (§4). 兩條路:手算、或 backprop。

*(Running example 貫穿例子: the [[04-neural-nets-ner]] window classifier — score $s=u^\top h$, $h=f(Wx+b)$,
$x$ = concatenated window word vectors.)*

## 2. Gradients by hand — matrix calculus 手算梯度:矩陣微積分

> "Multivariable calculus is just single-variable calculus **if you use matrices**." 用矩陣,多變數就跟單變數一樣。

### 2.1 Gradient & Jacobian 梯度與 Jacobian
- **1 output, $n$ inputs** → gradient = a **vector** of partials $\big[\frac{\partial f}{\partial x_1},\dots\big]$.
- **$m$ outputs, $n$ inputs** → **Jacobian** = an $m\times n$ matrix, $\big(\frac{\partial f}{\partial x}\big)_{ij}=\frac{\partial f_i}{\partial x_j}$. 一堆偏導排成矩陣。

### 2.2 Chain rule = multiply Jacobians 連鎖律 = Jacobian 相乘
For composed functions, **multiply** the Jacobians (order matters). 複合函數把 Jacobian 乘起來。 e.g. for
$h=f(z),\ z=Wx+b$: $\dfrac{\partial h}{\partial x}=\dfrac{\partial h}{\partial z}\dfrac{\partial z}{\partial x}$.

### 2.3 Key Jacobians to know 幾個要記的 Jacobian
| Function | Jacobian |
|---|---|
| **element-wise** $h=f(z)$ | $\operatorname{diag}\!\big(f'(z)\big)$ — a diagonal matrix; in practice just an element-wise $\odot f'(z)$. 對角矩陣。|
| $z=Wx+b$, w.r.t. $x$ | $W$ |
| $z=Wx+b$, w.r.t. $b$ | $I$ (identity) |
| $s=u^\top h$, w.r.t. $h$ | $u^\top$ |

($\odot$ = Hadamard / element-wise product. 逐元素相乘。)

## 3. Gradients of the neural net (the error signal δ) 神經網路的梯度

**The recipe 三步驟**: ① break the equations into simple pieces (track every shape!); ② apply the chain
rule; ③ write out the Jacobians. 拆式子(盯 shape)→ 連鎖律 → 寫 Jacobian。

### 3.1 The error signal δ — re-use computation 誤差訊號 δ
Computing $\partial s/\partial W$ and $\partial s/\partial b$ **share the same prefix** of the chain. Give
that shared upstream part a name — the **error signal / upstream gradient**:
$$\boldsymbol{\delta}=\frac{\partial s}{\partial z}=\underbrace{u^\top}_{\partial s/\partial h}\odot\underbrace{f'(z)}_{\partial h/\partial z}$$
Compute $\delta$ **once**, reuse it — that reuse is the whole idea of backprop. 算一次 δ 重複用,正是 backprop 精神。

### 3.2 Derivative w.r.t. a matrix + the shape convention 對矩陣微分 + shape 慣例
Pure math would make $\partial s/\partial W$ a $1\times nm$ Jacobian — awkward for SGD. So we use the
**shape convention**: **the gradient has the same shape as the parameter**. 梯度形狀 = 參數形狀。
$$\frac{\partial s}{\partial W}=\boldsymbol{\delta}\,x^\top\quad(n\times m,\ \text{an **outer product**}),\qquad
\frac{\partial s}{\partial b}=\boldsymbol{\delta}$$
- **Why the transpose / outer product** 為何是外積: each input feeds each output → $\delta$ (error at the
  output $z$) times $x$ (local input) gives every $W_{ij}$ contribution. 每個輸入接每個輸出 → 外積。
- **Sanity trick** 對照維度: if shapes don't match, transpose/reorder until they do — $\delta$ arriving at a
  layer has the **same shape as that layer**. δ 的形狀跟該層一樣,對不上就轉置。

## 4. ⭐ Backpropagation 反向傳播

Backprop = **the chain rule applied along a computation graph, reusing shared sub-derivatives**. 就是在
計算圖上套連鎖律、並重複利用共享的偏導。

### 4.1 Computation graph + forward pass 計算圖 + 前向
Software writes the net as a **graph**: source nodes = inputs, interior nodes = operations, edges pass
results. Running it left-to-right = **forward propagation** (save the intermediate values!). 前向:算出每個
節點的值並**存起來**。

### 4.2 Single node: the one rule 單一節點:唯一規則
Each node has a **local gradient** (output w.r.t. its input). Given the **upstream gradient**:
$$\boxed{[\text{downstream}] = [\text{upstream}] \times [\text{local}]}$$
下游梯度 = 上游梯度 × 本地梯度。 Multiple inputs → one downstream per input (each with its own local gradient).

### 4.3 Node intuitions 節點直覺 (+ · max · ×)
- **`+` distributes** 分配: copies the upstream gradient to **each** summand unchanged. 加法把梯度原封複製給每個加數。
- **`max` routes** 路由: sends the whole upstream gradient to the **arg-max** input, `0` to the others. 只給最大的那條。
- **`×` switches** 交換: each input's downstream = upstream × **the other** input's value. 乘法把梯度乘上「另一個輸入」。

### 4.4 Two efficiency rules 兩個效率要點
- **Gradients sum at outward branches** 分岔處相加: if a value is used in several places, **add** the
  gradients coming back from all of them. 一個值被用在多處 → 回傳的梯度相加。
- **Compute all gradients at once** (don't recompute shared prefixes) — exactly why we cached $\delta$. 一次算完,別重算。

### 4.5 The general algorithm 通用演算法
1. **Fprop**: visit nodes in **topological order**, compute each from its predecessors (save values). 拓樸序前向。
2. **Bprop**: initialize output gradient $=1$, visit nodes in **reverse order**, apply
   `downstream = upstream × local` (summing over successors). 反序、輸出梯度設 1、往回套規則。

Done right, **fprop and bprop have the same big-O cost**. 兩者複雜度同級。

## 5. In practice: autodiff & gradient checking 實務

- **Automatic differentiation** 自動微分: PyTorch/TensorFlow build the graph and do bprop for you — you
  only hand-write each node's **forward** and **local gradient**. 框架自動,你只寫節點的 forward + local grad。
- **Numeric gradient check** 數值梯度檢查 (for debugging a layer): $\frac{\partial J}{\partial\theta}\approx
  \frac{J(\theta+h)-J(\theta-h)}{2h}$ with $h\approx10^{-4}$. Correct but **slow** (recompute per parameter);
  use it only to verify an implementation. 準但慢,只拿來驗證。
- **Why learn this** 為何要懂: frameworks do it for you, but backprop **doesn't always work out of the box**
  — knowing the internals is key to debugging (e.g. **exploding / vanishing gradients**, a later lecture).
  懂內部才好 debug(梯度爆炸/消失)。

## 6. One-line summary 一句話總結

> **Training = SGD, which needs $\nabla_\theta J$.** Compute it with the **chain rule**; **backprop** does
> this efficiently on the computation graph via one rule — **downstream = upstream × local** — caching the
> shared **error signal δ** and summing gradients at branches. Forward saves values, backward applies the
> chain rule. 訓練靠 SGD 要梯度;backprop 用「下游=上游×本地」在計算圖上有效率地套連鎖律,快取 δ、分岔相加。
