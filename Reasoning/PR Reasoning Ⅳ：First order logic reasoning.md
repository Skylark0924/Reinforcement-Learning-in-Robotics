#! https://zhuanlan.zhihu.com/p/262984951
![Image](https://pic4.zhimg.com/80/v2-4ac8d686d2573c10f9fe4c4da3172801.jpg)
# PR Reasoning Ⅳ：数理逻辑（命题逻辑、谓词逻辑）知识整理
> 写完这个题目，我就想起了在犀安路999和 polimi 某量子大佬一起，被离散数学支配的恐惧。放在书架上的那本橙黄相间的古旧教科书，是唯一一本被我从本科带到现在的书，完全不记得当初用意何在。时隔三年再翻开这本落满灰尘的书，不禁感叹因果与宿命的力量。\
> 可能这就是人生吧。

![](https://pic4.zhimg.com/80/v2-ee933ec61bd5b4008bfb9b54faecce45.png)

## 基本逻辑符号表
> 转自 https://blog.csdn.net/lynn0085/article/details/87986813

![](https://pic4.zhimg.com/80/v2-e7881fdec1b9d9aff8b44170c80495d8.png)
![](https://pic4.zhimg.com/80/v2-65d6e42c71ff24d4ebecf6f7b923feaa.png)
![](https://pic4.zhimg.com/80/v2-387ef04f279acb36a1d768add366b5e0.png)

|     符号      |                   名字                    |                             解说                             |                             例子                             |                        读作                        |        范畴        |
| :-----------: | :---------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :------------------------------------------------: | :----------------: |
|      $→$      |    蕴含，实质蕴含 implies/conditional/    | $A → B$ 意味着如果 A 为真，则 B 也为真；如果 A 为假，则对 B 没有任何影响 | $x=2\rightarrow x^2=4$ 为真，但 $x^2=4\rightarrow x=2$ 一般为假，因为可以有 $x=-2$ |            仅为真值表蕴含式；如果…那么             |      命题逻辑      |
|      $⇒$      | 严格蕴含（模态逻辑） implies/conditional/ |           $A ⇒ B$ 表示不仅 A 蕴含 B ，而且内容相关           |                                                              |           严格蕴含，内容相关；如果…那么            |      模态逻辑      |
|      $↔$      |                 实质等价                  | $A ↔ B$ 意味着 $A$ 为真 则$B$ 为真，和 $A$ 为假 则 $B$ 为假。 |                       $x+5=y+2↔x+3=y$                        |                   当且仅当；iff                    |      命题逻辑      |
|      $⇔$      |           严格等价（模态逻辑）            |            $A ⇔ B$ ， $A$与$B$之间必须内容相关。             |                                                              |                   当且仅当；iff                    |      模态逻辑      |
|      $¬$      |                 逻辑否定                  |                  $¬A$ 为真，当且仅当 A 为假                  |                         $¬(¬A) ↔ A$                          |                         非                         |      命题逻辑      |
|      $∧$      |               逻辑**合取**                |      当A 与 B二者都为真，则陈述 $A ∧ B$ 为真；否则为假       |        $n < 4 ∧ n >2 ⇔ n = 3$（当 n 是自 然数的时候）        |                         与                         |      命题逻辑      |
|      $∨$      |               逻辑**析取**                | 当A 或 B有一个为真或二者均为真陈述，则 $A ∨ B$ 为真；当二者都为假，则 陈述为假。 |      $n ≣ 4 ∨ n ≢ 2 ⇔ n ≠ 3$（当 $n$ 是 自然数的时候）       |                         或                         |      命题逻辑      |
|      $∀$      |                 全称量词                  |     $∀ x: P(x)$ 意味着对所有的 $x$ 都使 $P(x)$ 都为真。      |                     $∀ n ∈ N（n² ≣ n）$                      |                 所有，每一个，任意                 |      谓词逻辑      |
|      $∃$      |                 存在量词                  |    $∃ x: P(x)$ 意味着有至少存在一个 $x$ 使 $P(x)$ 为真。     |                  $∃ n ∈ N$（$n$ 是偶数）。                   |                 存在着，至少有一个                 |      谓词逻辑      |
|     $∃!$      |                 唯一量词                  |     $∃! x: P(x)$ 意味着精确的有一个 $x$ 使 $P(x)$ 为真。     |                    ∃! n ∈ N（n + 5 = 2n）                    |                   精确的存在一个                   |      谓词逻辑      |
|  $\Psi(x)$   |                任意目谓词                 |      $\Psi : psi$，读音”普赛“ ，大写 $ \Psi$， 小写 $ψ$      |                 $\Psi()$是任意目谓词的元变项                 |        $\Psi(x)$ 代表任意目谓词构成的开语句        |      谓词逻辑      |
|    $\iota$    |  摹状词里用希腊字母ι \iota*ι* 代替定冠词  | $ \iota : iota$ ，读音”约塔“ 或者”艾欧塔“。大写 $\Iota$ ， 小写 $\iota$ | 摹状词结构：定冠词 the+形容词+名词单数，符号化为 ι $\iota xp （x）$ | $q( \iota xp (x))$读做：那个唯一具有性质p的个体是q |      谓词逻辑      |
|      $∵$      |                   因为                    |                                                              |                                                              |                                                    |                    |
|      $∴$      |                   所以                    |                                                              |                                                              |                                                    |                    |
|  $\square$   |                  模态词                   |                             必然                             |                              -                               |                        必然                        |         -          |
|  $\diamond$   |                  模态词                   |                             可能                             |                              -                               |                        可能                        |         -          |
|     $┌└┃$     |             推演过程流程符号              |                推演过程假设域需要用的流程符号                |                              -                               |                                                    |         -          |
|      $⊕$      |                    xor                    | 陈述 $A ⊕ B$ 为真，在要么 A 要么 B 但不是二者为真的时候为真。 |             $(¬A) ⊕ A$ 总是真，$A ⊕ A$ 总是假。              |                        异或                        | 命题逻辑，布尔代数 |
|      $/$      |                 命题逻辑                  |         穿过其他算符的斜线同于在它前面放置的"$¬$"。          |                      $x ≠ y ↔ ¬(x = y)$                      |                         非                         |      命题逻辑      |
| $:=$ 或者 $≡$ |                   定义                    | $x := y$ 或 $x ≡ y$ 意味着 $x$ 被定义为 $y$ 的另一个名字(但要注意 $≡$ 也可以意味着其他东西，比如全等)。 |      双曲余弦函数$cosh x := (1/2)(\exp x + \exp (−x))$       |                      被定义为                      |      所有地方      |
|     $:⇔$      |                   定义                    |         $P :⇔ Q$ 意味着 $P$ 被定义为逻辑等价于 $Q$。         |           $A \text{ XOR } B :⇔ (A ∨ B) ∧ ¬(A ∧ B)$           |                      被定义为                      |      所有地方      |
|      $├$      |                   推论                    |               $x ├ y$ 意味着 $y$ 推导自 $x$。                |                      $A → B ├  ¬B → ¬A$                      |                     推论或推导                     | 命题逻辑, 谓词逻辑 |
|      $├$      |                  断定符                   |                              -                               |                              -                               |                  (公式在L中可证)                   |         -          |
|      $╞$      |                  满足符                   |                              -                               |                              -                               |          (公式在E上有效，公式在E上可满足)          |         -          |


## 命题逻辑
### 基础知识
**命题**：一个非真即假( 不可兼) 的陈述句，或，只有具有确定真值(True or False)的陈述句才是命题。
> 例：雪是白的 / 雪是黑的。

- 原子命题：不包含任何的与、或、非一类联结词的命题，不能划分为更简单的陈述句。
- 复合命题：把一个或几个简单命题用联结词( 如与、或、非) 联结所构成的新的命题。

**联结词**：
- 否定：$\neg$\
  ![](https://pic4.zhimg.com/80/v2-35c51f5c73d9312c6d32f1eb89c37bd1.png)
- 合取：$\land$\
  ![](https://pic4.zhimg.com/80/v2-468bcedb33646807d69ea509b313fddd.png)
- 析取：$\lor$\
  ![](https://pic4.zhimg.com/80/v2-cb72d337bf284a3159a813177b7128aa.png)
- 条件：$\to$\
  ![](https://pic4.zhimg.com/80/v2-4066b663d61fe5c1beded4afa686b5ef.png)
- 双条件：$\leftrightarrow$\
  ![](https://pic4.zhimg.com/80/v2-031d1264fd0d784e44cdc53a97ecdc33.png)

**优先顺序**：$\neg, \land, \lor, \to, \leftrightarrow$


### 等值公式
![](https://pic4.zhimg.com/80/v2-3dfa05e84d9290360d2ddbed7b395bd4.png)
![](https://pic4.zhimg.com/80/v2-36c2601eb044c98c3709f7611ba8b7ec.png)
| 命题定律 |                            表达式                            | 序号 |
| :------: | :----------------------------------------------------------: | :--: |
|  对合律  |              $\neg \neg \mathrm{P}=\mathrm{P}$               |  1   |
|  冥等律  | $\begin{array}{l}P \vee P=P \\P \wedge P=P \\P \rightarrow P=T \\P \backslash P=T\end{array}$ |  2   |
|  结合律  | $\begin{array}{l}(\mathrm{P} \vee \mathrm{Q}) \vee \mathrm{R}=\mathrm{P} \vee(\mathrm{Q} \vee \mathrm{R}) \\(\mathrm{P} \wedge \mathrm{Q}) \wedge \mathrm{R}=\mathrm{P} \wedge(\mathrm{Q} \wedge \mathrm{R}) \\(\mathrm{P} \leftrightarrow \mathrm{Q}) \leftrightarrow \mathrm{R}=\mathrm{P} \leftrightarrow(\mathrm{Q} \leftrightarrow \mathrm{R}) \\(\mathrm{P} \rightarrow \mathrm{Q}) \rightarrow \mathrm{R} \neq \mathrm{P} \rightarrow(\mathrm{Q} \rightarrow \mathrm{R})\end{array}$ |  3   |
|  交换律  | $\begin{array}{l}\mathrm{P} \vee \mathrm{Q}=\mathrm{Q} \vee \mathrm{P} \\\mathrm{P} \wedge \mathrm{Q}=\mathrm{Q} \wedge \mathrm{P} \end{array}$ |  4   |
|  分配律  | $\begin{array}{l}P \vee(Q \wedge R)=(P \vee Q) \wedge(P \vee R) \\P \wedge(Q \vee R)=(P \wedge Q) V(P \wedge R) \\P \rightarrow(Q \rightarrow R)=(P \rightarrow Q) \rightarrow(P \rightarrow R) \\P \leftrightarrow(Q \leftrightarrow R) \neq(P \leftrightarrow Q) \leftrightarrow(P \leftrightarrow R)\end{array}$ |  5   |
|  吸收律  | $\begin{array}{l}P V(P \wedge Q)=P \\P \wedge(P \vee Q)=P\end{array}$ |  6   |
|  摩根律  | $\begin{array}{l}\neg(\mathrm{P} \vee \mathrm{Q})=\neg \mathrm{P} \wedge \neg \mathrm{Q} \\\neg(\mathrm{P} \wedge \mathrm{Q})=\neg \mathrm{P} \vee \neg \mathrm{Q}\end{array}$ |  7   |
|  同一律  | $\begin{aligned}&\mathrm{P} \vee \mathrm{F}=\mathrm{P}\\&\mathrm{P} \wedge \mathrm{T}=\mathrm{P}\\&\mathrm{T} \rightarrow \mathrm{P}=\mathrm{P}\\&\mathrm{T} \leftrightarrow \mathrm{P}=\mathrm{P}\end{aligned}$ |  8   |
|   零律   | $\begin{array}{l}\mathrm{P} \vee \mathrm{T}=\mathrm{T} \\\mathrm{P} \wedge \mathrm{F}=\mathrm{F} \\\mathrm{P} \rightarrow \mathrm{T}=\mathrm{T} \\\mathrm{F} \rightarrow \mathrm{P}=\mathrm{T}\end{array}$ |  9   |
|  否定律  | $\begin{array}{l}P \vee \neg P=T \\P \wedge \neg P=F \\P \rightarrow \neg P=\neg P \\\neg P \rightarrow P=P \\P \leftrightarrow  \neg P=F\end{array}$ |  10  |

### 重言式和蕴含式
**重言式**：如果一个公式, 对于它的任一解释 I 下其真值都为真, 就称
为重言式(永真公式)。

**矛盾式**：如果一个公式, 对于它的任一解释 I 下真值都是假, 便称是矛盾式(永假公式)。

**由∨、∧、→和\ 联结的重言式仍是重言式**

**蕴含式**：当且仅当 $P\rightarrow Q$ 是一个重言式时，我们称“P蕴含Q”，并记 $P\Rightarrow Q$ (P 推出 Q)

### 对偶与范式
**对偶式**：将A中出现的$∨、∧、T、F$分别以$∧、
∨、F、T$代换, 得到公式$A^*$
, 则称$A^*$是$A$的对偶式, 或
说$A^*$和$A$互为对偶式。

**范式**：与任何一个命题公式等值而形式不同的
命题公式可以有无穷多个，因此要将命题公司规范化。

**合取式**：一些文字的合取称为合取式

**析取式**：一些文字的析取称为析取式(也称子句)

**合取范式**： 一个命题公式称为合取范式，当且仅当它具有型式：
$$
A_1∧A_2∧……∧A_n，（n>=1）
$$
其中 $A_1, A_2, \dots, A_n$ 都是由命题变元或其否定所组成的**析取式**。（不唯一）

**析取范式**：一个命题公式称为析取范式,当且仅当它具有型式:
$$
A_1∨A_2∨……∨A_n，（n>=1）
$$
其中 $A_1, A_2, \dots, A_n$ 都是由命题变元或其否定所组成的**合取式**。（不唯一）

**范式定理**：任一命题公式都存在与之等值
的析取范式、合取范式

**求范式三步曲**：
1) 消去 $\rightarrow$ 和 $\leftrightarrow$
2) 否定深入到命题变项
3) 使用分配律的等值变换

![](https://pic4.zhimg.com/80/v2-439c832cc64cb1720a5beafc9967ea8d.png)
![](https://pic4.zhimg.com/80/v2-b79f658ec15ae3c283a568c4bccb1398.png)

![](https://pic4.zhimg.com/80/v2-fe70b111174155f7c08a784c476bc2fa.png)

**主合取范式**：仅由极大项构成的合取式。（唯一）
![](https://pic4.zhimg.com/80/v2-3234bba75c74c5d0095c4c91b79b12ca.png)
![](https://pic4.zhimg.com/80/v2-762e454f4d7e7e4260a8a84de9c347ce.png)

**主析取范式**：仅由极小项构成的析取式。（唯一）
![](https://pic4.zhimg.com/80/v2-a70cf2f548f25eb075486042506cd1b1.png)
![](https://pic4.zhimg.com/80/v2-0ffe9b78fdda422caf528b28953b58e5.png)

**主范式功能**
![](https://pic4.zhimg.com/80/v2-299dfa5557a9b5b0068f883b1ec382cb.png)
![](https://pic4.zhimg.com/80/v2-a2d774896f1902da8242ac29613c6255.png)
![](https://pic4.zhimg.com/80/v2-0167f741a7af3dd2f1ebc50582d34b64.png)

## 谓词逻辑
### 基础知识
从命题逻辑到谓词逻辑是一个更加细化的过程。命题逻辑中，最小组成单元是原子命题，是一个完整的句子。然而实际上，原子命题内部还包括更多的信息，如果忽略掉这些信息，会导致一些简单的论断无法推证。
> 例：所有人都要死，苏格拉底是人，所以苏格拉底是要死的。

因此就有了谓词逻辑。在谓词逻辑中，原子命题可分解成客体和谓词：
- **客体**：可独立存在的事或物；
- **谓词**：用以刻画**客体**的性质或关系。

谓 词 逻 辑 中 两 个 核 心 概 念 ： 谓 词 （ predicate ） 和 量 词 （quantifier）。量词分为：
- **存在量词**（existential quantifier) $\exists$
- **全称量词**(universal quantifier) $\forall$

## 一阶逻辑与高阶逻辑
> 在分析二者区别的诸多文章中，我终于找到了几句人话。

wiki上是这么说的：
- 一阶谓词演算或**一阶逻辑**（FOL）禁止量化谓词。

- **高阶逻辑**（HOL）亦称“广义谓词逻辑”、“高阶谓词逻辑”，可以出现更多变量类型的量化，更具表达性。高阶谓词是接受其他谓词作为参数的谓词。一般的，阶为n的高阶谓词接受一个或多个（n − 1）阶的谓词作为参数，这里的n > 1。对高阶函数类似的评述也成立。

具体的量化范围：
- 一阶逻辑仅量化范围广泛的变量；
- 二阶逻辑也可以量化集合；
- 三阶逻辑还对集合的集合进行量化，依此类推。
- 高阶逻辑是一阶，二阶，三阶，…，n阶逻辑的并集；即，高阶逻辑允许对任意深度嵌套的集合进行量化。

以一阶、二阶为例：
> 转自 辰述: https://www.cnblogs.com/chenshu/p/12634595.html

“对于任意的个体变项 x 和 y，如果 x 和 y 相等，那么对于任意性质或关系 F，F(x) 当且仅当 F(y)。”这在一阶逻辑中是无法表达出来的。因为一阶逻辑只能量化个体，而性质是包含个体的。但当我们引入了二阶逻辑后，就可以表达出来，这句话用二阶逻辑写出来会是这样：$∀x∀y( (x=y) → ∀F(F(x) ↔ F(y))$。从中我们可以看到，二阶逻辑可以量化包含个体词的集合（性质或关系），那么依次类推，更高阶的逻辑就是可以量化前一阶所能量化到的集合的集合。

