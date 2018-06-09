# analyzing-recommend-system
## Environment 
- Python 3.6 upper
- Numpy 1.4 upper

## 2018.05.16 update
- 矩阵奇异值分解（SVD）
    - [算法解析](https://github.com/ThomasHuai/Recommend/tree/master/matrix_factorization/svd)
    - [图片降噪](https://github.com/ThomasHuai/Recommend/tree/master/matrix_factorization/svd/image_denoising)
    - [Funk-SVD](https://github.com/ThomasHuai/Recommend/blob/master/matrix_factorization/svd/svd.pyx)

> 在推荐系统众多方法中，基于用户的协同过滤推荐算法是最早诞生的，原理也较为简单。该算法1992年提出并用于邮件过滤系统，两年后1994年被 GroupLens 用于新闻过滤。一直到2000年，该算法都是推荐系统领域最著名的算法。

俗话说“物以类聚、人以群分”，拿看电影这个例子来说，如果你喜欢《蝙蝠侠》、《碟中谍》、《星际穿越》、《源代码》等电影，另外有个人也都喜欢这些电影，而且他还喜欢《钢铁侠》，则很有可能你也喜欢《钢铁侠》这部电影。

所以说，当一个用户 A 需要个性化推荐时，可以先找到和他兴趣相似的用户群体 G，然后把 G 喜欢的、并且 A 没有听说过的物品推荐给 A，这就是基于用户的协同过滤算法。

根据上述基本原理，我们可以将基于用户的协同过滤推荐算法拆分为两个步骤：

1. 找到与目标用户兴趣相似的用户集合。
2. 找到这个集合中用户喜欢的、并且目标用户没有听说过的物品推荐给目标用户。
豆瓣电影是中国最著名的电影sns社区，它允许用户对每部电影进行评价。 


现在从豆瓣的用户中抽取了500左右个比较活跃的用户，这些用户都是忠实的电影迷，大部分人涉猎了上百部电影。

这里有个80多万行的[文本文件]("http://www.qlcoder.com/download/train.txt")，文件的每行是三个数字，分别是userid，movieid，rating。代表一个用户对一部电影的评分。rating代表评分的星级，如上图中的红框所示，星级从低到高依次是1-5。

接下来有个行数为10001的[文本文件]("http://www.qlcoder.com/download/test.txt")（第一行为title），文件的每行为2个数字，分别代表userid和movieid，请你预测如果该用户观看了这部电影，会给该电影打多少分，你的预测值为1个大小为1-5的整数。

本题的答案是一个长度为1万的字符串，字符串的第k位代表你对第k行的预测结果。

如果你的预测结果和实际答案的差值的绝对值的和小于6000，通过该题。


## run
```
python main.py
```

you get answer