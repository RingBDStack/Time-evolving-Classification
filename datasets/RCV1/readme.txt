
#0：
BD23:/storage2/heyu/data/data_Router


#1：
G:\ACT\1 本科毕设\data\RCV1\rcv1v2\ReutersCorpusVolume1\Data\ReutersCorpusVolume1_Original
RCV1的全部原始文档，1996-1997
BD25：/storage1/lyp/InputFiles/Data/ReutersCorpusVolume1_Original
GPU67：/root/heyu/Data/RCV1/RCV1/
RCV1的全部原始文档（解压后）

#2：
rcv1v2-ids.dat
RCV1所有文档的索引（文档ID），按照时间顺序从2286到810935，rcv1v2-ids.dat文件中的每一行xxx对应原始文档集合里面的一个文件xxx.xml。

#3：
BD25:/storage1/lyp/InputFiles/RouterGraph
BD23:/storage2/heyu/data/data_Router/graphs
文件夹内是与文档ID对应的每个文档的graph文件（不包含label）
如2286,xml对应的graph文件是2286.graph

#4：
rcv1-v2.topics.qrels
RCV1各个文档的topic查询query，即这个文件按照#2的索引，给出了各个文件的label。
而且这个label是扩展后的：这个label列表，已经按照层次依赖的父子关系进行了扩展。

#5：
rcv1.topics.txt
103个topic列表，行数为索引，所以就是103个label的索引文件。

#6：
rcv1.topics.hier.orig
103个label的父子关系，是原始文件，即topics的父子关系，不是remap之后的。

#7：
fathercode
103个label的父子关系，相比#6作了一次映射，直接给出了这些label的索引的父亲的索引是哪个

#8：
docs
RCV1的全部文档集合，原始的xml文件解析之后得到的纯文本信息。（去除停用词以及词干化之后的结果）

#8：
labels:
RCV1的全部文档所对应的标签索引。














