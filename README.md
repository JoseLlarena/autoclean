# Autoclean

Autoclean is a set of tools to clean textual data with minimal human intervention:

`autoclean.filtering` uses unsupervised learning to filter out of domain sequences
`autoclean.segmentation` uses unsupervised learning to segment joined up sequences

## Filtering

The filtering package contains both an api and a cli.

An example use through the cli to filter out out-of-domain/anomalous data points:

1. Download an English book from Project Gutenberg
```shell
(venv) $ wget  https://www.gutenberg.org/cache/epub/29440/pg29440.txt
--2022-02-15 00:14:04--  https://www.gutenberg.org/cache/epub/29440/pg29440.txt
Resolving www.gutenberg.org (www.gutenberg.org)... 152.19.134.47, 2610:28:3090:3000:0:bad:cafe:47
Connecting to www.gutenberg.org (www.gutenberg.org)|152.19.134.47|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 890028 (869K) [text/plain]
Saving to: ‘pg29440.txt’

pg29440.txt                                         100%[==================================================================================================================>] 869.17K   730KB/s    in 1.2s    

2022-02-15 00:14:06 (730 KB/s) - ‘pg29440.txt’ saved [890028/890028]
```
2. Run the filtering giving a high cost-threshold, so the output is just all the lines but ordered in decreasing order of 
cost:
```shell
(venv)$ time python -m autoclean.filtering.cli filter pg29440.txt filtered.pg20440.txt 10000
[2022-02-15 00:26:38][INFO][autoclean._estimate:178] > estimating [6]-gram from [pg29440.txt] ...
[2022-02-15 00:26:40][INFO][autoclean._estimate:182] > binarising [6]-gram from [/tmp/pg29440.txt.2022-02-15T00:26:38.2265355794898408859.6.arpa] ...
[2022-02-15 00:26:40][INFO][autoclean._estimate:178] > estimating [1]-gram from [pg29440.txt] ...
[2022-02-15 00:26:40][INFO][autoclean._estimate:182] > binarising [1]-gram from [/tmp/pg29440.txt.2022-02-15T00:26:40.2265355794898408859.1.arpa] ...
[2022-02-15 00:26:40][INFO][autoclean.filtering.filter_out: 30] > Reading from [pg29440.txt] and writing to [filtered.pg20440.txt] ...
[2022-02-15 00:26:40][INFO][autoclean.filtering.filter_out: 43] > Done.

real	0m3.201s
user	0m1.882s
sys		0m2.803s
```
3. Check the top of the output file for standard English sentences:
```shell
(venv) $ head -n25 filtered.pg20440.txt 
in some
of the country?
of the whole:--
according to the poet,
Election--Responsibility--Influence of the Church in such a
in the original.
character of the schoolmaster, and the right and duty of the
there of the influence which the personal character of the
their Church is the Church of the nation, and that it is they
true, and in the just and the true only.
stream of not half the volume of that in which the money of the
savageism of the other.
class with which the schoolmaster and the class with which the
between the feelings of the people and the anticipations of some of
the light of the flames.'
in any age of the Presbyterian Church, in one of the parish
According to the poet,
circumstances of Scotland in his days and of Scotland in the present
the State is the money of the people, and that the people have a right
is not according to the nature of things that the case of the tenant
their part of the scheme.'
which all recognition of the religious element on the part of the
simply those of the subject and the ratepayer.
character of our poor Highlanders, and of the influence of the bothie
themselves, and not the ministers of the Establishment, who are on
```
3. Check the bottom of the output file for non-standard English text:
```shell
(venv) $ tail -n25 filtered.pg20440.txt
residence.'
periodicals.
consistency.
predecessor.
differently?
FINE-BODYISM.
ELIGIBLE.'{8}
Glencalvie:--
photographer.
resuscitated.
PERIODICALISM.
double-quote}
relinquished.
circumstances.
Protestantism.
significantly,
impossibility.
responsibility.
redistribution.
gbnewby@pglaf.org
kingdom.'--SISMONDI.
self-recommendation.
http://www.gutenberg.org
http://gutenberg.org/license).
http://www.gutenberg.org/2/9/4/4/29440/
```

The cli can also be used to evaluate the filtering algorithm with an in-domain and an out-of-domain text:

1. Download French book from Project Gutenberg
```shell
(venv) $ wget  https://www.gutenberg.org/cache/epub/46916/pg46916.txt
--2022-02-15 00:41:00--  https://www.gutenberg.org/cache/epub/46916/pg46916.txt
Resolving www.gutenberg.org (www.gutenberg.org)... 152.19.134.47, 2610:28:3090:3000:0:bad:cafe:47
Connecting to www.gutenberg.org (www.gutenberg.org)|152.19.134.47|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 120716 (118K) [text/plain]
Saving to: ‘pg46916.txt’

pg46916.txt                                         100%[==================================================================================================================>] 117.89K   347KB/s    in 0.3s    

2022-02-15 00:41:01 (347 KB/s) - ‘pg46916.txt’ saved [120716/120716]
```
2. Run evaluation on the Downloaded English and French books, using the former as in-domain and the latter as out-of-domain
```shell
(venv) $ time python -m autoclean.filtering.cli eval pg29440.txt pg46916.txt 
[2022-02-15 00:47:29][INFO][autoclean.filtering.evaluate: 62] > Reading from [pg29440.txt] and to [pg46916.txt] ...
[2022-02-15 00:47:29][INFO][autoclean._estimate:178] > estimating [6]-gram from [/tmp/pg29440.txt.pg46916.txt.2022-02-15T00:47:29.1382531328938857964.txt] ...
[2022-02-15 00:47:31][INFO][autoclean._estimate:182] > binarising [6]-gram from [/tmp/pg29440.txt.pg46916.txt.2022-02-15T00:47:29.1382531328938857964.txt.2022-02-15T00:47:29.6537006284435656167.6.arpa] ...
[2022-02-15 00:47:31][INFO][autoclean._estimate:178] > estimating [1]-gram from [/tmp/pg29440.txt.pg46916.txt.2022-02-15T00:47:29.1382531328938857964.txt] ...
[2022-02-15 00:47:31][INFO][autoclean._estimate:182] > binarising [1]-gram from [/tmp/pg29440.txt.pg46916.txt.2022-02-15T00:47:29.1382531328938857964.txt.2022-02-15T00:47:31.6537006284435656167.1.arpa] ...
[2022-02-15 00:47:31][INFO][autoclean.filtering.evaluate: 74] > Calculating 10%-cutoff rankings metrics ...

@100% precision [ 94.6832%] recall [ 94.6832%] fallout [ 38.4824%]

@ 90% precision [ 96.7801%] recall [ 87.1050%] fallout [ 20.9756%]

@ 80% precision [ 97.4352%] recall [ 77.9467%] fallout [ 14.8509%]

@ 70% precision [ 97.6359%] recall [ 68.3466%] fallout [ 11.9783%]

@ 60% precision [ 97.8532%] recall [ 58.7090%] fallout [  9.3225%]

@ 50% precision [ 98.1878%] recall [ 49.0939%] fallout [  6.5583%]

@ 40% precision [ 98.4650%] recall [ 39.3889%] fallout [  4.4444%]

@ 30% precision [ 98.4024%] recall [ 29.5192%] fallout [  3.4688%]

@ 20% precision [ 98.5399%] recall [ 19.7095%] fallout [  2.1138%]

@ 10% precision [ 98.4270%] recall [  9.8397%] fallout [  1.1382%]

[2022-02-15 00:47:32][INFO][autoclean.filtering.evaluate:104] > Done.

real	0m3.400s
user	0m1.967s
sys	    0m2.622s
```

## Segmentation

Segmentation is based on the recursive dynamic-programming algorithm presented by Peter Norvig in 
[chapter 14 of "Beautiful Data"](http://norvig.com/ngrams/ch14.pdf), by Seagaran and Hammerbacher. 

It features an api and cli. An example usage through the cli:

1. Download file to use as corpus (complete works of shakespeare)
```shell
(venv) $ wget http://norvig.com/ngrams/shakespeare.txt
--2022-02-15 12:47:22--  http://norvig.com/ngrams/shakespeare.txt
Resolving norvig.com (norvig.com)... 158.106.138.13
Connecting to norvig.com (norvig.com)|158.106.138.13|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 4538523 (4.3M) [text/plain]
Saving to: ‘shakespeare.txt’

shakespeare.txt                                     100%[==================================================================================================================>]   4.33M  3.29MB/s    in 1.3s    

2022-02-15 12:47:24 (3.29 MB/s) - ‘shakespeare.txt’ saved [4538523/4538523]
```
2. Create the file to be segmented by deleting spaces between words:
```shell
(venv) $ sed  's/\s//g' shakespeare.txt > joined.shakespeare.txt
```
3. Run the segmentation with a sixgram language model:
```shell
(venv) $ time python -m autoclean.segmentation.cli seg --lm=5 -c shakespeare.txt joined.shakespeare.txt seg.shakespeare.txt
[2022-02-15 12:48:35][INFO][autoclean.segmentation.segment: 29] > Reading in corpus from [shakespeare.txt] ...
[2022-02-15 12:48:35][INFO][autoclean.segmentation.segment: 42] > Estimating [6]-ngram language model from [shakespeare.txt] ...
[2022-02-15 12:48:35][INFO][autoclean._estimate:178] > estimating [6]-gram from [/tmp/corpus.2022-02-15T12:48:35.6.txt] ...
[2022-02-15 12:48:44][INFO][autoclean._estimate:182] > binarising [6]-gram from [/tmp/corpus.2022-02-15T12:48:35.6.txt.2022-02-15T12:48:35.3998131699779813468.6.arpa] ...
[2022-02-15 12:48:45][INFO][autoclean.segmentation.segment: 45] > Segmenting and saving to [seg.shakespeare.txt] ...
[2022-02-15 12:48:45][INFO][autoclean.segmentation.segment: 49] > Reading in file to segment from [joined.shakespeare.txt] ...
[2022-02-15 12:52:56][INFO][autoclean.segmentation.segment: 59] > Done.

real	4m32.020s
user	4m30.212s
sys	0m2.493s
```
4. Evaluate the segmentation using the original text as a gold standard:

```shell
(venv) $ time python -m autoclean.segmentation.cli eval shakespeare.txt seg.shakespeare.txt 
[2022-02-15 12:55:32][INFO][autoclean.segmentation.evaluate: 70] > Reading segmented text from [shakespeare.txt] ...
[2022-02-15 12:55:32][INFO][autoclean.segmentation.evaluate: 71] > Reading gold standard text from [seg.shakespeare.txt] ...
[2022-02-15 12:55:32][INFO][autoclean.segmentation.evaluate: 84] > Accuracy: [99.80%], [128,844] correct out of [129,107] sequences
[2022-02-15 12:55:32][INFO][autoclean.segmentation.evaluate: 87] > Done.

real	0m0.624s
user	0m0.732s
sys	0m0.723s
```

# Installation

As Autoclean is on Pypi and has direct dependencies like Pytorch and KenLM, and Pypi doesn't support them,
they need to be installed separately and previous to installing Autoclean: 

```shell
$ git clone git@github.com:JoseLlarena/autoclean.git
$ python3 virtualenv -p=3.8 venv
$ source venv/bin/activate
(venv) $ pip3 install -f https://download.pytorch.org/whl/cu113/torch_stable.html -r ./autoclean/requirements.txt 
```
Then install Autoclean with pip:
```shell
(venv) $ pip3 install autoclean
```

The code requires python 3.8+ on and Nvidia GPU. The KenLM/SRILM binaries have been compiled against Linux x86_64 and may 
not work on other architectures.
 

## Testing

The unit tests are written with [pytest](https://docs.pytest.org/en/stable). Run with:

```shell
$ pip3 install pytest

$ pytest
```

## Changelog

Check the [Changelog](https://github.com/JoseLlarena/autoclean/blob/master/CHANGELOG.md) for fixes and enhancements of
each version.

## License

Copyright Jose Llarena 2022.

Distributed under the terms of the [MIT](https://github.com/JoseLlarena/autoclean/blob/master/LICENSE) license,
autoclean is free and open source software.