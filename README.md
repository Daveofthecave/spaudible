# Spaudible
![Python 3.12](https://img.shields.io/badge/python-3.12-64b128?style=plastic)
![Lines of Code](https://aschey.tech/tokei/github/Daveofthecave/spaudible?category=lines&style=plastic&color=64b128)
![File Count](https://aschey.tech/tokei/github/Daveofthecave/spaudible?category=files&style=plastic&color=64b128)
![Repo Size](https://img.shields.io/github/repo-size/Daveofthecave/spaudible?style=plastic&color=64b128)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=plastic&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+CiAgICA8Zz4KICAgICAgICA8cGF0aCBmaWxsPSJub25lIiBkPSJNMCAwSDI0VjI0SDB6Ii8+CiAgICAgICAgPHBhdGggZD0iTTMgNS40NzlsNy4zNzctMS4wMTZ2Ny4xMjdIM1Y1LjQ4em0wIDEzLjA0Mmw3LjM3NyAxLjAxN3YtNy4wNEgzdjYuMDIzem04LjE4OCAxLjEyNUwyMSAyMXYtOC41MDJoLTkuODEydjcuMTQ4em0wLTE1LjI5MnY3LjIzNkgyMVYzbC05LjgxMiAxLjM1NHoiIGZpbGw9IiNmZmZmZmYiLz4KICAgIDwvZz4KPC9zdmc+&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=plastic&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KCjwhLS0gTGludXggU1ZHIC0tPgo8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKYXJpYS1sYWJlbD0iTGludXgiIHJvbGU9ImltZyIKdmlld0JveD0iMCAwIDUxMiA1MTIiIGZpbGw9IiMwMDAwMDAiPgoKPGcgdHJhbnNmb3JtPSJtYXRyaXgoMi42NSAwIDAgMi42NSAyNTYgMjc2KSI+Cgo8cGF0aCBkPSJNLTMyLTI1Yy0zIDctMjQgMjktMjIgNTEgOCA5MiAzNiAzMCA3OCA1MyAwIDAgNzUtNDIgMTUtMTEwLTE3LTI0LTItNDMtMTMtNTlzLTMwLTE3LTQ0LTIgNiAzNy0xNCA2NyIvPgoKPHBhdGggZD0iTTQyIDIxczktMTgtOC0zMWMxNiAxNyA2IDMyIDYgMzJoLTNDMzYtMTMgMjcgNiAxNC01NiAyOS03MyAwLTg4IDAtNjBoLTljMS0yNC0yMC0xMi04IDUtMSAzNy0yMyA1Mi0yMyA3OC03LTE4IDYtMzIgNi0zMnMtMTggMTUtNyAzNyAzMSAxNyAxNyAyN2MyMiAxNSA1NiA1IDU1LTI3IDEtOCAyMi01IDI0LTNzLTMtNC0xMy00bS01Ni03OGMtNy0yLTUtMTEtMi0xMXM4IDcgMiAxMW0xOSAxYy01LTctMS0xNCA0LTEzczUgMTMtNCAxMyIgZmlsbD0iI2ZmZiIvPgoKPGcgZmlsbD0iI2U5YzEzMiIgc3Ryb2tlPSIjMzMzIiBzdHJva2Utd2lkdGg9IjEiPgoKPHBhdGggZD0iTS00MSAzMWwyMSAzMGMxMSA3IDUgMzUtMjUgMjEtMTctNS0zMS00LTMzLTEzczQtMTAgMy0xNGMtNC0yMiAxNC0xMSAxOS0yMnM1LTE2IDE1LTJNNzEgNDVjLTQtNiAwLTE3LTE0LTE2LTYgMTItMjMgMjQtMjQgMC0xMCAwLTMgMjQtNyAzNS05IDI3IDE3IDI5IDI4IDE2bDI2LTE4YzItMyA1LTYtOS0xN20tOTItOTJjLTMtNiAxMS0xNCAxNi0xNHMxMiA0IDE5IDYgNCA5IDIgMTBTMy0zNS01LTM1cy0xMC04LTE2LTEyIi8+Cgo8cGF0aCBkPSJNLTIxLTQ4YzggNiAxNyAxMSAzNS0zIi8+Cgo8L2c+Cgo8cGF0aCBkPSJNLTEwLTU0Yy0yIDAgMS0yIDItMW03IDFjMS0xLTEtMi0zLTEiLz4KCjwvZz4KCjwvc3ZnPg==&logoColor=black)
![macOS](https://img.shields.io/badge/macOS-555555?style=plastic&logo=apple&logoColor=white)


Spaudible is an offline search engine that helps you discover new music acoustically similar to your favorite songs. It achieves this by converting your input song into a mathematical vector (i.e. a song "fingerprint"), comparing this vector against a quarter billion other song vectors, and returning a playlist of songs that scored the highest in similarity.

<p align="center">
  <img src="https://raw.githubusercontent.com/wiki/Daveofthecave/spaudible/assets/spaudible-search-16fps.gif" height="450">
</p>

On a modern PC with an Nvidia GPU, finding similar songs takes only a few seconds. This is possible thanks to the CUDA-accelerated PyTorch library, which rapidly computes vector similarities in parallel across a custom-built vector cache. Even on systems without Nvidia hardware, Spaudible falls back to an efficient Numba-accelerated CPU pipeline that completes similarity searches within minutes.

## How It Works

Every song has its own set of unique attributes; for example:

- Genre (eg. rock, classical, jazz)
- Tempo (eg. 120 bpm)
- Key (eg. C major, F♯ minor)
- Release year (eg. 2007)

One way to determine how similar two songs are to each other is to compare their attributes. If two songs share similar genres, tempos, keys, release years, and other attributes, we can conclude that those songs are indeed similar. On the contrary, if two songs have completely different genres, tempos, etc., we can conclude that they aren't very similar at all.

We humans naturally compare songs in a similar way, although these comparisons usually happen subconsciously and instantaneously. Hard rock "_feels_" very different from classical music, and a fast tempo "_feels_" more danceable than a slow tempo.

A computer, however, only understands numbers. If we'd want the computer to determine how similar two songs are, we would need to translate the attributes of those songs into its native language – numbers. This translation process is called "encoding".

Some song attributes are easier to encode than others. Tempo and release year, for instance, already exist as numbers, which makes things a lot easier. Genre and key, on the other hand, are usually represented with words, letters, and/or symbols. Translating these non-numeric attributes to numbers in a meaningful way is slightly trickier. That being said, there are many reasonable ways to do so. 

<details>
<summary>Example</summary>

>In Spaudible's case, for example, it groups musical keys together by the number of accidentals (sharps or flats), from 0 to 6, and then divides by 6 to obtain a normalized value between 0 and 1. This way, keys with fewer sharps or flats get a value closer to 0, and keys with many sharps or flats get a value closer to 1. Of course, there are many other ways to encode a key (eg. chromatically, by circle of fifths, etc.), but Spaudible uses this particular encoding to avoid overfitting while maintaining enharmonic synergy.

</details>

Once we figure out how we want to encode our song attributes, we need an efficient way to store them. One of the most common ways of storing a group of similar items is in a list. Computer scientists often like to use the term "**array**", while mathematicians are more partial to the term "**vector**", but in the end, both terms describe a collection of items.

That being said, while there may be overlap in semantics, there remain subtle (and often not-so-subtle) distinctions among certain terms that may help us in tackling our similarity problem. 

Thinking of a list of numbers as a "vector" carries with it certain mathematical connotations that turn out to be useful for us. In math, a vector is a special list of numbers that has both direction and magnitude (length). Think of it like a container that stores numbers in an organized way, one after the other. Just like we can perform arithmetic and algebra on individual numbers, we can also do math on vectors – groups of numbers. This is essentially what **linear algebra** is; algebra on a "line" of numbers.

We can visualize vectors as arrows of varying lengths pointing in particular directions.

<p align="center">
  <img src="https://raw.githubusercontent.com/wiki/Daveofthecave/spaudible/assets/cosine-demo.gif" height="330">
</p>

Some vectors point in similar directions, while others point in opposite directions. We can tell how closely aligned two vectors are by calculating their **cosine similarity**. In other words, what is the **cosine** of the angle between them? Cosine behaves like a percentage. It can tell you, for example, that "vector _b_ is 95% aligned with vector _a_."

Turns out this is pretty useful for our song comparison problem! Since we've determined that we can convert the attributes of any song into a vector, this means we take advantage of these powerful vector operations to determine how similar two vectorized songs are!

This is exactly what Spaudible does under the hood.

Using a preprocessed binary cache of song vectors, in tandem with Spotify's databases that hold attributes to over 256 million songs, Spaudible takes a song the user provides, converts it into a vector, and then applies cosine similarity (or a related algorithm) between the user's song vector and the quarter billion other vectors sitting in the cache. Rather than going through the entire vector cache one-by-one, Spaudible splits up this job across the CPU's cores, or, better yet, across the GPU's more numerous CUDA cores. Since it doesn't matter in what order we conduct our similarity calculations, we can leverage the power of parallel processing to reach our final result faster.





