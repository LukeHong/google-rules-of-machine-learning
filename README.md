# Google's *43 Rules of Machine Learning*

Github mirror of [M. Zinkevich's](http://martin.zinkevich.org/)  great ["Rules of Machine Learning"](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf) style guide, with extra goodness.

-----

You can find the terminology for this guide in [terminology.md](/terminology.md).

You can find the overview for this guide in [overview.md](/overview.md).

#### Structure

1. [Before Machine Learning](#before-machine-learning)
2. [ML Phase 1: Your First Pipeline](#your-first-pipeline)
3. [ML Phase 2: Feature Engineering](#feature-engineering)
4. [ML Phase 3: Slow Growth, Optimation Refinement, and Complex Models](#slow-growth-and-optimization-and-complex-models)
5. [Related Work](/related_work.md)
6. [Acknowledgements & Appendix](/acknowledgements_and_appendix.md)

**Note**: *Asterisk* (\*) footnotes are my own. *Numbered* footnotes are Martin's.

## Before Machine Learning

#### Rule 1 - Don't be afraid to launch a product without machine learning.*
Machine learning is cool, but it requires data. Theoretically, you can take data from a different problem and then tweak the model for a new product, but this will likely underperform basic heuristics. If you think that machine learning will give you a 100% boost, then a heuristic will get you 50% of the way there. For instance, if you are ranking apps in an app marketplace, you could use the install rate or number of installs. If you are detecting spam, filter out publishers that have sent spam before. Don’t be afraid to use human editing either. If you need to rank contacts, rank the most recently used highest (or even rank alphabetically). If machine learning is not absolutely required for your product, don't use it until you have data.

<sup>[Google Research Blog - The 280-Year-Old Algorithm Inside Google Trips](https://research.googleblog.com/2016/09/the-280-year-old-algorithm-inside.html?m=1)</sup>

#### Rule 2 - First, design and implement metrics.
Before formalizing what your machine learning system will do, track as much as possible in your current system. Do this for the following reasons:

1. It is easier to gain permission from the system’s users earlier on.
2. If you think that something might be a concern in the future, it is better to get historical
data now.
3. If you design your system with metric instrumentation in mind, things will go better for
you in the future. Specifically, you don’t want to find yourself grepping for strings in logs
to instrument your metrics!
4. You will notice what things change and what stays the same.

For instance, suppose you want to directly optimize one-­day active users. However, during your early manipulations of the system, you may notice that dramatic alterations of the user experience don’t noticeably change this metric.
Google Plus team measures expands per read, reshares per read, plus­-ones per read, comments/read, comments per user, reshares per user, etc. which they use in computing the goodness of a post at serving time. Also, note that an experiment framework, where you can group users into buckets and aggregate statistics by experiment, is important. See Rule **#12**.

By being more liberal about gathering metrics, you can gain a broader picture of your system. Notice a problem? Add a metric to track it! Excited about some quantitative change on the last
release? Add a metric to track it!

#### Rule 3 - Choose machine learning over complex heuristic.

A simple heuristic can get your product out the door. A complex heuristic is unmaintainable. Once you have data and a basic idea of what you are trying to accomplish, move on to machine
learning. As in most software engineering tasks, you will want to be constantly updating your approach, whether it is a heuristic or a machine-learned model, and you will find that the
machine-­learned model is easier to update and maintain (see Rule **#16**).

## Your First Pipeline

> Focus on your system infrastructure for your first pipeline. While it is fun to think about all the
imaginative machine learning you are going to do, it will be hard to figure out what is happening
if you don’t first trust your pipeline.

#### Rule 4 - Keep the first model simple and get the infrastructure right.

The first model provides the biggest boost to your product, so it doesn't need to be fancy. But you will run into many more infrastructure issues than you expect. Before anyone can use your
fancy new machine learning system, you have to determine:

1. How to get examples to your learning algorithm.
2. A first cut as to what “good” and “bad” mean to your system.
3. How to integrate your model into your application. You can either apply the model live, or pre­compute the model on examples offline and store the results in a table. For example,
you might want to pre­classify web pages and store the results in a table, but you might want to classify chat messages live.

Choosing simple features makes it easier to ensure that:

1. The features reach your learning algorithm correctly.
2. The model learns reasonable weights.
3. The features reach your model in the server correctly.

Once you have a system that does these three things reliably, you have done most of the work. Your simple model provides you with baseline metrics and a baseline behavior that you can use
to test more complex models. Some teams aim for a “neutral” first launch: a first launch that explicitly de-­prioritizes machine learning gains, to avoid getting distracted.

#### Rule 5 - Test the infrastructure independently from the machine learning.

Make sure that the infrastructure is testable, and that the learning parts of the system are
encapsulated so that you can test everything around it. Specifically:

1. Test getting data into the algorithm. Check that feature columns that should be populated
are populated. Where privacy permits, manually inspect the input to your training algorithm. If possible, check statistics in your pipeline in comparison to elsewhere, such
as RASTA.

2. Test getting models out of the training algorithm. Make sure
that the model in your
training environment gives the same score as the model in your serving environment (see Rule **#37**). Machine learning has an element of unpredictability, so make sure that you have tests for the code for creating examples in training and serving, and that you can load and use a fixed model during serving. Also, it is important to understand your data: see [Practical Advice for Analysis of Large, Complex Data Sets.](http://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html)

#### Rule 6 - Be careful about dropped data when copying pipelines.

Often we create a pipeline by copying an existing pipeline (i.e. [cargo cult programming](https://en.wikipedia.org/wiki/Cargo_cult_programming)), and the old pipeline drops data that we need for the new pipeline. For example, the pipeline for Google
Plus What’s Hot drops older posts (because it is trying to rank fresh posts). This pipeline was copied to use for Google Plus Stream, where older posts are still meaningful, but the pipeline
was still dropping old posts. Another common pattern is to only log data that was seen by the user. Thus, this data is useless if we want to model why a particular post was not seen by the
user, because all the negative examples have been dropped. A similar issue occurred in Play. While working on Play Apps Home, a new pipeline was created that also contained examples
from two other landing pages (Play Games Home and Play Home Home) without any feature to disambiguate where each example came from.

#### Rule 7 - Turn heuristics into features, or handle them externally.

Usually the problems that machine learning is trying to solve are not completely new. There is an existing system for ranking, or classifying, or whatever problem you are trying to solve. This means that there are a bunch of rules and heuristics. These same heuristics can give you a lift when tweaked with machine learning. Your heuristics should be mined for whatever information they have, for two reasons. First, the transition to a machine learned system will be smoother. Second, usually those rules contain a lot of the intuition about the system you don’t want to throw away. There are four ways you can use an existing heuristic:

1. Preprocess using the heuristic. If the feature is incredibly awesome, then this is an option. For example, if, in a spam filter, the sender has already been blacklisted, don’t try
to relearn what “blacklisted” means. Block the message. This approach makes the most sense in binary classification tasks.
2. Create a feature. Directly creating a feature from the heuristic is great. For example, if you use a heuristic to compute a relevance score for a query result, you can include the score as the value of a feature. Later on you may want to use machine learning techniques to massage the value (for example, converting the value into one of a finite
set of discrete values, or combining it with other features) but start by using the raw value produced by the heuristic.
3. Mine the raw inputs of the heuristic. If there is a heuristic for apps that combines the number of installs, the number of characters in the text, and the day of the week, then
consider pulling these pieces apart, and feeding these inputs into the learning separately. Some techniques that apply to ensembles apply here (see **Rule #40**).
4. Modify the label. This is an option when you feel that the heuristic captures information not currently contained in the label. For example, if you are trying to maximize the number of downloads, but you also want quality content, then maybe the solution is to multiply the label by the average number of stars the app received. There is a lot of space here for leeway. See the section on [“Your First Objective”](#your-first-objective). Do be mindful of the added complexity when using heuristics in an ML system. Using old heuristics in your new machine learning algorithm can help to create a smooth transition, but think about whether there is a simpler way to accomplish the same effect.

### 監控 (Monitoring)

> 一般來說，使一個警告通知包含可執行的動作及監控報表頁面，是不錯的實踐。

#### Rule 8 - 了解你的系統對時效性的需求

如果你有個一天前的模型，會降低多少的效能表現？一周前？一月前呢？這項資訊可以幫助你了解監控的優先度。
如果這個模型一天不更新就會損失 10% 的營收，那麼讓一個工程師持續監控就是一件合理的事。

大多數的廣告服務系統每天都有新的廣告要處理，因此必須每天更新。舉例來說，如果 Google Play 商店的搜尋功能沒更新，在一個月內就會衝擊到營收。
某些 Google+ 熱門內容的模型並不包含貼文編號，因此可以偶爾再送出新模型，而某些包含貼文編號的模型就必須更經常的更新。

值得注意的是，時效性會隨著時間改變，特別是當特徵欄位被從模型中加入或移除時。

#### Rule 9 - 在送出模型之前發現問題

許多機器學習系統會有一個送出模型至服務的階段，如果送出的模型有問題，就會是一個影響到使用者的問題 (userfacing issue)。
但如果發生在送出之前，就只是一個訓練上的問題 (training issue)，而不會影響到使用者。

在部屬模型之前要做合理性確認 (sanity check)，特別是確保在給出的資料之下模型的表現是合理的。如果對於資料抱持懷疑，就不要部屬這個模型。
許多的團隊在持續部屬模型之前，會檢查 [ROC 曲線](https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF) 下方的面積 (或是 [AUC](http://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it))。

還沒部屬出去的模型出問題時，只需要電子郵件的警告通知，但是影響到使用者的模型可能需要有監控頁面。因此在接觸使用者之前，最好等到確認完畢為止。

#### Rule 10 - 小心沉默的錯誤 (silent failures)

比起其他種類的系統，這是個更常發生於機器學習系統的問題。

例如某個使用到的資料表已經停止更新了，機器學習系統會適應並且有合理的好表現，但是漸漸地崩壞。有時會發現一些資料表的資料已經過期了數個月，比許多其他嘗試，只要簡單的更新資料就能提升更多的效能表現。

某個特徵的涵蓋度可能會因實作上的更動而改變，例如某個特徵欄位本來存在於 90% 的樣本中，但突然掉到剩 60% 的樣本中。Play 商店曾經有個過期了六個月的資料表，只更新這個資料表就提升了 2% 的安裝率。如果你追蹤資料的統計數據或是不時的手動查看資料，就可以減少這類型的錯誤。*

* <sup> [*A Framework for Analysis of Data Freshness* - Bouzeghoub & Peralta](https://www.fing.edu.uy/inco/grupos/csi/esp/Publicaciones/2004/iqis2004-mb.pdf)</sup>

#### Rule 11 - 把特徵欄位指定一個負責人以及建立文件 Give feature columns owners and documentation.

當系統龐大且有著許多特徵欄位時，需要知道每個欄位由誰建立或維護。當你發現某個了解一個特徵欄位的人要離開時，確保有人能交接資訊。
縱然許多欄位有明確的名字，但如果有關於這特徵為何、從何而來及預期的效果為何的詳細敘述會更好。

### 你的第一個目標 (Objective)

> 對於你在意的系統，有著許多的指標 (metrics) 與測量 (measurements)，但是你的機器學習演算法通常需要一個 **單一的目標，一個你的演算法「嘗試」去最佳化的的數字**。我在此區分指標跟目標的不同：**指標是任何你系統反應出來的數字**，可能重要也可能不重要。可以參閱規則 **#2**

#### Rule 12 - 不要過度考慮你要最佳化哪個目標

你想要賺錢、讓使用者開心、讓世界變得更美好。有著大量的指標可以去注意，而且你應該去測量所有的指標（見規則 **#2**）。
但是在機器學習過程的早期，你會發現所有都提升了，就算不是你想要最佳化的。

例如你關心著點擊數、在網站上的停留時間以及每日活躍用戶，當你提升點擊數時，你很有可能會發現停留時間也增加了。
所以當你還能輕易的提升所有指標時，別太過於思考如何平衡各項指標。但也別太過頭：不要用系統的 ultimate health 混淆了你的目標（見規則 #39），以及 **如果你發現你提升了想要最佳化的指標，但決定不要不要發佈，那可能需要重新檢視目標**。

#### Rule 13 - 為你的第一個目標選擇一個簡單、可觀察及可歸因的指標 Choose a simple, observable and attributable metric for your first objective.

常常你以為你知道什麼是真正的目標，直到你盯著你的舊系統與新機器學習系統的比較數據及分析時，你才發現到你想修正一下。進一步來說，團隊成員間經常無法在真正的目標上達成共識。機器學習的目標應該是可以簡單測量的「真正」目標的替代品。

So train on the simple ML objective, and consider having a "policy layer" on top that allows you to add additional logic (hopefully very simple logic) to do the final ranking.

The easiest thing to model is a user behavior that is directly observed and attributable to an
action of the system:

1. Was this ranked link clicked?
2. Was this ranked object downloaded?
3. Was this ranked object forwarded/replied to/e­mailed?
4. Was this ranked object rated?
5. Was this shown object marked as spam/pornography/offensive?

Avoid modeling indirect effects at first:

1. Did the user visit the next day?
2. How long did the user visit the site?
3. What were the daily active users?
Indirect effects make great metrics, and can be used during A/B testing and during launch
decisions.

Finally, don’t try to get the machine learning to figure out:

1. Is the user happy using the product?
2. Is the user satisfied with the experience?
3. Is the product improving the user’s overall well­being?
4. How will this affect the company’s overall health?

These are all important, but also incredibly hard. Instead, use proxies: if the user is happy, they will stay on the site longer. If the user is satisfied, they will visit again tomorrow. Insofar as well­being and company health is concerned, human judgement is required to connect any machine learned objective to the nature of the product you are selling and your business plan, so we don’t end up [here](https://www.youtube.com/watch?v=bq2_wSsDwkQ).

#### Rule 14 - 從一個可解釋（interpretable）的模型讓偵錯更容易 Starting with an interpretable model makes debugging easier.

[線性回歸](https://en.wikipedia.org/wiki/Linear_regression), [邏輯回歸](https://en.wikipedia.org/wiki/Logistic_regression), and [帕松回歸](https://en.wikipedia.org/wiki/Poisson_regression) are directly motivated by a probabilistic model. Each prediction is interpretable as a probability or an expected value. This makes them easier to debug than models that use objectives (zero­one loss, various hinge losses, et cetera) that try to directly optimize classification accuracy or ranking performance. For example, if probabilities in training deviate from probabilities predicted in side­-by-­sides or by
inspecting the production system, this deviation could reveal a problem.

For example, in linear, logistic, or Poisson regression, **there are subsets of the data where the average predicted expectation equals the average label (1­moment calibrated, or just calibrated)<sup>3</sup>**. If you have a feature which is either 1 or 0 for each example, then the set of examples where that feature is 1 is calibrated. Also, if you have a feature that is 1 for every example, then the set of all examples is calibrated.

With simple models, it is easier to deal with feedback loops (see Rule **#36&**). Often, we use these probabilistic predictions to make a decision: e.g. rank posts in decreasing expected value (i.e. probability of click/download/etc.). However, remember when it comes time to choose which model to use, the decision matters more than the likelihood of the data given the model (see Rule **#27**).

#### Rule 15 - Separate Spam Filtering and Quality Ranking in a Policy Layer.

Quality ranking is a fine art, but spam filtering is a war.\* The signals that you use to determine high quality posts will become obvious to those who use your system, and they will tweak their posts to have these properties. Thus, your quality ranking should focus on ranking content that is posted in good faith. You should not discount the quality ranking learner for ranking spam highly. **Similarly, “racy” content should be handled separately from Quality Ranking.** Spam filtering is a different story. You have to expect that the features that you need to generate will be constantly changing. Often, there will be obvious rules that you put into the system (if a
post has more than three spam votes, don’t retrieve it, et cetera). Any learned model will have to be updated daily, if not faster. The reputation of the creator of the content will play a great role.

At some level, the output of these two systems will have to be integrated. Keep in mind, filtering
spam in search results should probably be more aggressive than filtering spam in email messages. Also, it is a standard practice to remove spam from the training data for the quality
classifier.

<sup>[Google Research Blog - Lessons learned while protecting Gmail](https://research.googleblog.com/2016/03/lessons-learned-while-protecting-gmail.html?m=1)</sup>

## 特徵工程 Feature engineering

> 在機器學習系統生命週期的第一階段中，重要的工作是將訓練資料送到系統中、選擇有興趣的目標、以及建立一個服務的架構。 **當你有個包含單元測試與系統測試的運作中 end to end 系統，第二階段就開始了**

#### Rule 16 - 有計劃的發佈與迭代 Plan to launch and iterate.

不要假設目前正在做的模型會是最終發佈的，或甚至不會再發佈新的模型。
因此要考慮這次增加的複雜度是否會拖慢之後新模型的發佈。

許多團隊每季或是更長的時間才發佈新的模型，有三個基本的原因去發佈新的模型：

1. 你有增加新的特徵
2. 你調整了正規化（regularization）以及用新方法組合舊的特徵
3. 以及／或是你調整了目標

不管如何，對一個模型投注更多心力會更好。looking over the data feeding into the example can help find new signals as well as old, broken ones.
所以當你建立模型時，要考慮新增、移除或重組特徵的難易度，考慮新建立一份複製的流程以及驗證正確性的難度，考慮擁有兩三個複製的流程平行運作的可能性。
Finally, don’t worry about whether feature 16 of 35 makes it into this version of the pipeline. You’ll get it next quarter.

#### Rule 17 - 從可直接觀察、被呈現的特徵而不是 "learned features" Start with directly observed and reported features as opposed to learned features.

這可能是一個有爭議的點，但它避免了不少陷阱。
首先來解釋一下什麼是 "learned features"？
"learned features" 是一個由外部系統（例如一個非監督式的分群系統）或是該機器學習系統本身（例如透過分解模型或是深度學習）產生的特徵。
這兩個都可能很有用，但是又有著許多的問題，因此不應該被用在剛開始的模型中。

如果你使用外部系統來產生一個特徵，要記得這個系統有他自己的目標，而跟你當前的目標可能只有微弱的關聯。
如果你使用了這個外部系統的快照（snapshot），則可能會變得過時。
當你更新了來自這外部系統的特徵時，它的意義可能會改變。
如果你使用一個外部系統來提供特徵，要知道這會需要大量額外的關注。

分解模型或深度模型的主要問題在於它們是 non­-convex，因此無法保證可以找到近似或是最佳解決方案，以及每次迭代找到的區域最佳解可能都不一樣。
這變化使其很難判斷一個改動造成的影響是有意義的，又或者只是隨機的。
藉由建立一個不使用深度特徵（deep features）的模型，可以得到極好的基礎效能表現。
達成了這個基礎以後，就可以嘗試更加深奧的方法。

#### Rule 18 - Explore with features of content that generalize across contexts.

機器學習系統常常只是更廣闊大局中的一小部分。
例如如果你想像一篇貼文可能被用在「熱門文章」中，許多人會在其顯示為「熱門文章」之前就按讚、分享或是在底下發表回應。
如果你把這些統計數據送去學習，它可以在其試著最佳化的情境中，推廣一篇沒有相關資料的新貼文。
例如 Youtube 的「即將播放」可以使用從搜尋來的觀看數或是接著觀看數（看完一個影片後接著看這影片的數量），你也可以直接使用使用者評分。

最後如果你把一個使用者行為作為 label，觀察這行為在不同情境的結果可以作為一個良好的特徵，這些特徵都可以讓你將新的內容帶入這情境中。
要記得這不是為了個人化：首先找出如果有人在這情境中喜歡這內容，接著找出誰喜歡這個多一點或少一點。

#### Rule 19 - 盡可能使用非常明確的特徵 Use very specific features when you can.

在擁有大量資料的狀況下，使用無數簡單特徵訓練會比使用一點點複雜的特徵訓練更簡單。
Identifiers of documents being retrieved and canonicalized queries do not provide much
generalization, but align your ranking with your labels on head queries.
因此不要怕使用一組每個都只對應到少數樣本，但整體涵蓋超過九成資料的特徵，你可以用正規化來排除對應到過少樣本的特徵。

#### Rule 20 - Combine and modify existing features to create new features in human-understandable ways.

There are a variety of ways to combine and modify features. Machine learning systems such as TensorFlow allow you to pre­process your data through [transformations](https://www.tensorflow.org/tutorials/linear/overview#feature-columns-and-transformations). The two most standard approaches are “discretizations” and “crosses”.

Discretization consists of taking a continuous feature and creating many discrete features from it. Consider a continuous feature such as age. You can create a feature which is 1 when age is less than 18, another feature which is 1 when age is between 18 and 35, et cetera. Don’t overthink the boundaries of these histograms: basic quantiles will give you most of the impact. Crosses combine two or more feature columns. A feature column, in TensorFlow's terminology, is a set of homogenous features, (e.g. {male, female}, {US, Canada, Mexico}, et cetera). A cross is a new feature column with features in, for example, *{male, female} × {US,Canada, Mexico}*. This new feature column will contain the feature (male, Canada). If you are using TensorFlow and you tell TensorFlow to create this cross for you, this (male, Canada) feature will be present
in examples representing male Canadians. Note that it takes massive amounts of data to learn models with crosses of three, four, or more base feature columns.

Crosses that produce very large feature columns may overfit. For instance, imagine that you are doing some sort of search, and you have a feature column with words in the query, and you
have a feature column with words in the document. You can combine these with a cross, but you will end up with a lot of features (see Rule **#21**). When working with text there are two
alternatives. The most draconian is a dot product. A dot product in its simplest form simply counts the number of common words between the query and the document. This feature can
then be discretized. Another approach is an intersection: thus, we will have a feature which is present if and only if the word “pony” is in the document and the query, and another feature
which is present if and only if the word “the” is in the document and the query.

#### Rule 21 - The number of feature weights you can learn in a linear model is roughly proportional to the amount of data you have.

There are fascinating statistical learning theory results concerning the appropriate level of complexity for a model, but this rule is basically all you need to know. I have had conversations in which people were doubtful that anything can be learned from one thousand examples, or that you would ever need more than 1 million examples, because they get stuck in a certain method of learning. The key is to scale your learning to the size of your data:

1. If you are working on a search ranking system, and there are millions of different words in the documents and the query and you have 1000 labeled examples, then you should use a dot product between document and query features, [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), and a half-­dozen other highly human-­engineered features. 1000 examples, a dozen features.
2. If you have a million examples, then intersect the document and query feature columns, using regularization and possibly feature selection. This will give you millions of features,
but with regularization you will have fewer. Ten million examples, maybe a hundred thousand features.
3. If you have billions or hundreds of billions of examples, you can cross the feature columns with document and query tokens, using feature selection and regularization. You will have a billion examples, and 10 million features.

Statistical learning theory rarely gives tight bounds, but gives great guidance for a starting point.
In the end, use Rule **#28** to decide what features to use.

#### Rule 22 - 清除不再使用的特徵 Clean up features you are no longer using.

不再使用的會造成技術債。當你發現你不再使用一個特徵，以及拿它與其他特徵組合也沒有用時，那就把它從架構中丟棄。
你想要保持架構的乾淨，如此一來最有用的特徵嘗試起來可以越快，必要時也有人可以把它加回架構中。

當要新增或保留一些特徵時，要記住它的涵蓋率。
這個特徵涵蓋了多少樣本？例如當你有一些個人化的特徵，但只涵蓋了 8% 的使用者，這就不是高效率的特徵。
同時有些特徵可能超乎預期的價值。例如有個特徵只涵蓋了 1% 的資料，但其中的 90% 都是有效的樣本，這就會是個適合增加的特徵。

### 系統的人為分析 Human Analysis of the System

> 在進入機器學習的第三階段之前，需要專注在一個任何機器學習課程都不會教的事：「如何觀察一個既有的模型並且改善它」。比起科學這更像是門藝術，但有些反面模式應該避免。

#### Rule 23 - You are not a typical end user.*

This is perhaps the easiest way for a team to get bogged down. While there are a lot of benefits to fish-fooding (using a prototype within your team) and dog-fooding (using a prototype within your company), employees should look at whether the performance is correct. While a change which is obviously bad should not be used, anything that looks reasonably near production should be tested further, either by paying laypeople to answer questions on a crowdsourcing platform, or through a live experiment on real users. There are two reasons for this. The first is that you are too close to the code. You may be looking for a particular aspect of the posts, or you are simply too emotionally involved (e.g. confirmation bias). The second is that your time is too valuable. Consider the cost of 9 engineers sitting in a one hour meeting, and think of how many contracted human labels that buys on a crowdsourcing platform.

If you really want to have user feedback, **use user experience methodologies**. Create user personas (one description is in Bill Buxton’s [~~Designing~~ *Sketching User Experiences*](https://www.amazon.com/Sketching-User-Experiences-Interactive-Technologies/dp/0123740371)) early in a process and
do usability testing (one description is in Steve Krug’s [*Don’t Make Me Think*](https://www.amazon.com/Dont-Make-Me-Think-Usability/dp/0321344758)) later. User personas involve creating a hypothetical user. For instance, if your team is all male, it might help to design a 35­-year old female user persona (complete with user features), and look at the results it generates rather than 10 results for 25­-40 year old males. Bringing in actual people to watch their reaction to your site (locally or remotely) in usability testing can also get you a fresh perspective.

<sup>[Google Research Blog - How to measure translation quality in your user interfaces](https://research.googleblog.com/2015/10/how-to-measure-translation-quality-in.html?m=1)

#### Rule 24 - 測量模型間的差異 Measure the delta between models

在任何使用者看到你的新模型之前，其中一個最簡單以及有時候最有效的測量方法，就是計算新的結果與 production 上的差了多少。
舉例來說，如果你有個排名問題（ranking problem），同時使用兩個模型來跑來自整個系統中的一筆樣本，看兩個結果的對稱差（symmetric difference）有多少（按照排名加權）。
如果差異非常小，你不需要進行實驗就能說這只會有微小變動。
如果差異非常巨大，則你就需要去確認是良性變動。
檢查對稱差很高的部分，可以幫助你有效的了解變動的是什麼。
然而需要確定這系統是穩定的，確保一個模型跟自己比較起來只有很低的對稱差（理想上是零）。

#### Rule 25 - 當選擇模型時，實際的表現更勝於預測 When choosing models, utilitarian performance trumps predictive power.

你的模型可能嘗試要預測點擊率（click­-through-­rate），然而最後的關鍵問題是你用這預測來做什麼。
如果你用它來排名文件，則最後排名的品質比預測本身重要。
如果你要預測這文件是垃圾訊息的機率然後阻擋下來，則最後通過的準確率更重要。

大多數時候這兩個東西應該是一致同意的，如果不是的話通常只會有微小收穫。
因此如果有些變動改善了 loss 但是降低了系統的表現，那就去檢查其他特徵。
當這開始越來越常發生時，就是重新考慮這模型目標的時候了。

#### Rule 26 - Look for patterns in the measured errors, and create new features.

Suppose that you see a training example that the model got “wrong”. In a classification task, this could be a false positive or a false negative. In a ranking task, it could be a pair where a positive was ranked lower than a negative. The most important point is that this is an example that the
machine learning system knows it got wrong and would like to fix if given the opportunity. If you give the model a feature that allows it to fix the error, the model will try to use it.
On the other hand, if you try to create a feature based upon examples the system doesn’t see as mistakes, the feature will be ignored. For instance, suppose that in Play Apps Search,
someone searches for “free games”. Suppose one of the top results is a less relevant gag app. So you create a feature for “gag apps”. However, if you are maximizing number of installs, and people install a gag app when they search for free games, the “gag apps” feature won’t have the effect you want.

Once you have examples that the model got wrong, look for trends that are outside your current feature set. For instance, if the system seems to be demoting longer posts, then add post
length. Don’t be too specific about the features you add. If you are going to add post length, don’t try to guess what long means, just add a dozen features and the let model figure out what to do with them (see Rule **#21**). That is the easiest way to get what you want.

#### Rule 27 - Try to quantify observed undesirable behavior.

Some members of your team will start to be frustrated with properties of the system they don’t like which aren’t captured by the existing loss function. At this point, they should do whatever it takes to turn their gripes into solid numbers. For example, if they think that too many “gag apps” are being shown in Play Search, they could have human raters identify gag apps. (You can feasibly use human-­labelled data in this case because a relatively small fraction of the queries account for a large fraction of the traffic.) If your issues are measurable, then you can start using them as features, objectives, or metrics. The general rule is **“measure first, optimize second”**.

#### Rule 28 - Be aware that identical short-term behavior does not imply identical long-term behavior.

Imagine that you have a new system that looks at every doc_id and exact_query, and then calculates the probability of click for every doc for every query. You find that its behavior is
nearly identical to your current system in both side by sides and A/B testing, so given its simplicity, you launch it. However, you notice that no new apps are being shown. Why? Well,
since your system only shows a doc based on its own history with that query, there is no way to learn that a new doc should be shown.

The only way to understand how such a system would work long­term is to have it train only on data acquired when the model was live. This is very difficult.

### Training-Serving Skew

> Training­-serving skew is a difference between performance during training and performance
during serving. This skew can be caused by:
* a discrepancy between how you handle data in the training and serving pipelines, or
* a change in the data between when you train and when you serve, or
* a feedback loop between your model and your algorithm.

> We have observed production machine learning systems at Google with training-­serving skew
that negatively impacts performance. The best solution is to explicitly monitor it so that system
and data changes don’t introduce skew unnoticed.

#### Rule 29 - The best way to make sure that you train like you serve is to save the set of features used at serving time, and then pipe those features to a log to use them at training time.

Even if you can’t do this for every example, do it for a small fraction, such that you can verify the consistency between serving and training (see Rule **#37**). Teams that have made this measurement at Google were sometimes surprised by the results. YouTube home page switched to logging features at serving time with significant quality improvements and a reduction in code complexity, and many teams are switching their infrastructure as we speak.

#### Rule 30 - Importance weight sampled data, don't arbitrarily drop it!

When you have too much data, there is a temptation to take files 1­12, and ignore files 13­99. This is a mistake: dropping data in training has caused issues in the past for several teams (see Rule **6**). Although data that was never shown to the user can be dropped, importance weighting is best for the rest. Importance weighting means that if you decide that you are going
to sample example X with a 30% probability, then give it a weight of 10/3. With importance weighting, all of the calibration properties discussed in Rule **#14** still hold.

#### Rule 31 - Beware that if you join data from a table at training and serving time, the data in the table may change.

Say you join doc ids with a table containing features for those docs (such as number of comments or clicks). Between training and serving time, features in the table may be changed.
Your model's prediction for the same document may then differ between training and serving. The easiest way to avoid this sort of problem is to log features at serving time (see Rule **#32**). If the table is changing only slowly, you can also snapshot the table hourly or daily to get reasonably close data. Note that this still doesn’t completely resolve the issue.

#### Rule 32 - Re-use code between your training pipeline and your serving pipeline whenever possible.

Batch processing is different than online processing. In online processing, you must handle each request as it arrives (e.g. you must do a separate lookup for each query), whereas in batch
processing, you can combine tasks (e.g. making a join). At serving time, you are doing online processing, whereas training is a batch processing task. However, there are some things that
you can do to re­use code. For example, you can create an object that is particular to your system where the result of any queries or joins can be stored in a very human readable way,
and errors can be tested easily. Then, once you have gathered all the information, during serving or training, you run a common method to bridge between the human-­readable object
that is specific to your system, and whatever format the machine learning system expects. **This eliminates a source of training-­serving skew.** As a corollary, try not to use two different programming languages between training and serving ­ that decision will make it nearly impossible for you to share code.

#### Rule 33 - If you produce a model based on the data until January 5th, test the model on the data from January 6th and after.

In general, measure performance of a model on the data gathered after the data you trained the model on, as this better reflects what your system will do in production. If you produce a model based on the data until January 5th, test the model on the data from January 6th. You will expect that the performance will not be as good on the new data, but it shouldn’t be radically worse. Since there might be daily effects, you might not predict the average click rate or conversion rate, but the area under the curve, which represents the likelihood of giving the positive example a score higher than a negative example, should be reasonably close.

#### Rule 34 - In binary classification for filtering (such as spam detection or determining interesting e­mails), make small short­term sacrifices in performance for very clean data.

In a filtering task, examples which are marked as negative are not shown to the user. Suppose you have a filter that blocks 75% of the negative examples at serving. You might be tempted to
draw additional training data from the instances shown to users. For example, if a user marks an
email as spam that your filter let through, you might want to learn from that. But this approach introduces sampling bias. You can gather cleaner data if instead during
serving you label 1% of all traffic as “held out”, and send all held out examples to the user. Now your filter is blocking at least 74% of the negative examples. These held out examples can
become your training data. Note that if your filter is blocking 95% of the negative examples or more, this becomes less
viable. Even so, if you wish to measure serving performance, you can make an even tinier sample (say 0.1% or 0.001%). Ten thousand examples is enough to estimate performance quite
accurately.

#### Rule 35 - Beware of the inherent skew in ranking problems.

When you switch your ranking algorithm radically enough that different results show up, you have effectively changed the data that your algorithm is going to see in the future. This kind of skew will show up, and you should design your model around it. There are multiple different approaches. These approaches are all ways to favor data that your model has already seen.

1. Have higher regularization on features that cover more queries as opposed to those features that are on for only one query. This way, the model will favor features that are
specific to one or a few queries over features that generalize to all queries. This approach can help prevent very popular results from leaking into irrelevant queries. Note
that this is opposite the more conventional advice of having more regularization on feature columns with more unique values.
2. Only allow features to have positive weights. Thus, any good feature will be better than a feature that is “unknown”.
3. Don’t have document-only features. This is an extreme version of #1. For example, even if a given app is a popular download regardless of what the query was, you don’t want to
show it everywhere<sup>4</sup>. Not having document-only features keeps that simple.

<sup>4 - The reason you don’t want to show a specific popular app everywhere has to do with the importance of
making all the desired apps reachable. For instance, if someone searches for “bird watching app”, they
might download “angry birds”, but that certainly wasn’t their intent. Showing such an app might improve
download rate, but leave the user’s needs ultimately unsatisfied.</sup>

#### Rule 36 - Avoid feedback loops with positional features.

The position of content dramatically affects how likely the user is to interact with it. If you put an app in the first position it will be clicked more often, and you will be convinced it is more likely to be clicked. One way to deal with this is to add positional features, i.e. features about the position of the content in the page. You train your model with positional features, and it learns to weight, for example, the feature "1st­position" heavily. Your model thus gives less weight to other factors for examples with "1st­position=true". Then at serving you don't give any instances the positional feature, or you give them all the same default feature, because you are scoring candidates before you have decided the order in which to display them. Note that it is important to keep any positional features somewhat separate from the rest of the
model because of this asymmetry between training and testing. Having the model be the sum of a function of the positional features and a function of the rest of the features is ideal. For example, don’t cross the positional features with any document feature.

#### Rule 37 - Measure training/serving skew.

There are several things that can cause skew in the most general sense. Moreover, you can divide it into several parts:

1. The difference between the performance on the training data and the holdout data. In general, this will always exist, and it is not always bad.
2. The difference between the performance on the holdout data and the “next­day” data. Again, this will always exist. **You should tune your regularization to maximize the next­day performance.** However, large drops in performance between holdout and next­day data may indicate that some features are time-­sensitive and possibly degrading
model performance.
3. The difference between the performance on the “next­day” data and the live data. If you apply a model to an example in the training data and the same example at serving, it
should give you exactly the same result (see Rule **#5**). Thus, a discrepancy here probably indicates an engineering error.

## Slow Growth and Optimization and Complex models

> There will be certain indications that the second phase is reaching a close. First of all, your monthly gains will start to diminish. You will start to have tradeoffs between metrics: you will see some rise and others fall in some experiments. This is where it gets interesting. Since the gains
are harder to achieve, the machine learning has to get more sophisticated. A caveat: this section has more blue-­sky rules than earlier sections. We have seen many teams
go through the happy times of Phase I and Phase II machine learning. Once Phase III has been reached, teams have to find their own path.

#### Rule 38 - Don't waste time on new features if unaligned objectives have become the issue.

As your measurements plateau, your team will start to look at issues that are outside the scope of the objectives of your current machine learning system. As stated before, if the product goals are not covered by the existing algorithmic objective, you need to change either your objective
or your product goals. For instance, you may optimize clicks, plus-­ones, or downloads, but make launch decisions based in part on human raters.

#### Rule 39 - Launch decisions are a proxy for long-term product goals.

Alice has an idea about reducing the logistic loss of predicting installs. She adds a feature. The
logistic loss drops. When she does a live experiment, she sees the install rate increase. However, when she goes to a launch review meeting, someone points out that the number of
daily active users drops by 5%. The team decides not to launch the model. Alice is disappointed, but now realizes that launch decisions depend on multiple criteria, only some of
which can be directly optimized using ML. The truth is that the real world is not dungeons and dragons: there are no “hit points” identifying the health of your product. The team has to use the statistics it gathers to try to effectively
predict how good the system will be in the future. They need to care about engagement, 1 day active users (DAU), 30 DAU, revenue, and advertiser’s return on investment. These metrics that are measureable in A/B tests in themselves are only a proxy for more long­term goals: satisfying users, increasing users, satisfying partners, and profit, which even then you could consider proxies for having a useful, high quality product and a thriving company five years from now.

**The only easy launch decisions are when all metrics get better (or at least do not get worse).** If the team has a choice between a sophisticated machine learning algorithm, and a
simple heuristic, if the simple heuristic does a better job on all these metrics, it should choose the heuristic. Moreover, there is no explicit ranking of all possible metric values. Specifically, consider the following two scenarios:

| Experiment | Daily Active Users | Revenue/Day |
|------------|--------------------|-------------|
| A          | 1 million          | $4 million  |
| B          | 2 million          | $2 million  |

If the current system is A, then the team would be unlikely to switch to B. If the current system is B, then the team would be unlikely to switch to A. This seems in conflict with rational behavior: however, predictions of changing metrics may or may not pan out, and thus there is a large risk involved with either change. Each metric covers some risk with which the team is concerned. Moreover, no metric covers the team’s ultimate concern, “where is my product going to be five
years from now”?

**Individuals, on the other hand, tend to favor one objective that they can directly optimize**. Most machine learning tools favor such an environment. An engineer banging out new features
can get a steady stream of launches in such an environment. There is a type of machine learning, multi­-objective learning, which starts to address this problem. For instance, one can
formulate a constraint satisfaction problem that has lower bounds on each metric, and optimizes some linear combination of metrics. However, even then, not all metrics are easily framed as machine learning objectives: if a document is clicked on or an app is installed, it is because that the content was shown. But it is far harder to figure out why a user visits your site. How to predict the future success of a site as a whole is [AI­complete](https://en.wikipedia.org/wiki/AI-complete), as hard as computer vision or
natural language processing.

#### Rule 40 - Keep ensembles simple.

Unified models that take in raw features and directly rank content are the easiest models to debug and understand. However, an ensemble of models (a “model” which combines the scores of other models) can work better. **To keep things simple, each model should either be an ensemble only taking the input of other models, or a base model taking many features,
but not both.** If you have models on top of other models that are trained separately, then combining them can result in bad behavior.

Use a simple model for ensembling that takes only the output of your “base” models as inputs.
You also want to enforce properties on these ensemble models. For example, an increase in the score produced by a base model should not decrease the score of the ensemble. Also, it is best
if the incoming models are semantically interpretable (for example, calibrated) so that changes of the underlying models do not confuse the ensemble model. **Also, enforce that an increase in the predicted probability of an underlying classifier does not decrease the predicted
probability of the ensemble.**

#### Rule 41 - When performance plateaus, look for qualitatively new sources of information to add rather than refining existing signals.

You’ve added some demographic information about the user. You've added some information about the words in the document. You have gone through template exploration, and tuned the
regularization. You haven’t seen a launch with more than a 1% improvement in your key metrics in a few quarters. Now what?
It is time to start building the infrastructure for radically different features, such as the history of documents that this user has accessed in the last day, week, or year, or data from a different property. Use [wikidata](https://en.wikipedia.org/wiki/Wikidata) entities or something internal to your company (such as Google’s [knowledge graph](https://en.wikipedia.org/wiki/Knowledge_Graph)). Use deep learning. Start to adjust your expectations on how much return you expect on investment, and expand your efforts accordingly. As in any engineering project, you have to weigh the benefit of adding new features against the cost of increased complexity.

#### Rule 42 - Don't expect diversity, personalization, or relevance to be as correlated with popularity as you think they are.

Diversity in a set of content can mean many things, with the diversity of the source of the content being one of the most common. Personalization implies each user gets their own results. Relevance implies that the results for a particular query are more appropriate for that query than any other. Thus all three of these properties are defined as being different from the ordinary.
The problem is that the ordinary tends to be hard to beat.

Note that if your system is measuring clicks, time spent, watches, +1s, reshares, et cetera, you are measuring the popularity of the content. Teams sometimes try to learn a personal model with diversity. To personalize, they add features that would allow the system to personalize (some features representing the user’s interest) or diversify (features indicating if this document has any features in common with other documents returned, such as author or content), and find that those features get less weight (or sometimes a different sign) than they expect. This doesn’t mean that diversity, personalization, or relevance aren’t valuable.\* As pointed out in the previous rule, you can do post-­processing to increase diversity or relevance. If you see longer term objectives increase, then you can declare that diversity/relevance is valuable, aside from popularity. You can then either continue to use your post­-processing, or directly modify the objective based upon diversity or relevance.

<sup>[Google Research Blog - App Discovery With Google Play](https://research.googleblog.com/2016/12/app-discovery-with-google-play-part-2.html?m=1)

#### Rule 43 - Your friends tend to be the same across different products. Your interests tend not to be.

Teams at Google have gotten a lot of traction from taking a model predicting the closeness of a connection in one product, and having it work well on another. Your friends are who they are. On the other hand, I have watched several teams struggle with personalization features across product divides. Yes, it seems like it should work. For now, it doesn’t seem like it does. What has sometimes worked is using raw data from one property to predict behavior on another. Also, keep in mind that even knowing that a user has a history on another property can help. For instance, the presence of user activity on two products may be indicative in and of itself.
