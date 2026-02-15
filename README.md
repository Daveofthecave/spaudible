# Spaudible
![Python 3.12](https://img.shields.io/badge/python-3.12-64b128?style=plastic)
![Lines of Code](https://aschey.tech/tokei/github/Daveofthecave/spaudible?category=lines&style=plastic&color=64b128)
![File Count](https://aschey.tech/tokei/github/Daveofthecave/spaudible?category=files&style=plastic&color=64b128)
![Repo Size](https://img.shields.io/github/repo-size/Daveofthecave/spaudible?style=plastic&color=64b128)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=plastic&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+CiAgICA8Zz4KICAgICAgICA8cGF0aCBmaWxsPSJub25lIiBkPSJNMCAwSDI0VjI0SDB6Ii8+CiAgICAgICAgPHBhdGggZD0iTTMgNS40NzlsNy4zNzctMS4wMTZ2Ny4xMjdIM1Y1LjQ4em0wIDEzLjA0Mmw3LjM3NyAxLjAxN3YtNy4wNEgzdjYuMDIzem04LjE4OCAxLjEyNUwyMSAyMXYtOC41MDJoLTkuODEydjcuMTQ4em0wLTE1LjI5MnY3LjIzNkgyMVYzbC05LjgxMiAxLjM1NHoiIGZpbGw9IiNmZmZmZmYiLz4KICAgIDwvZz4KPC9zdmc+&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=plastic&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KCjwhLS0gTGludXggU1ZHIC0tPgo8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKYXJpYS1sYWJlbD0iTGludXgiIHJvbGU9ImltZyIKdmlld0JveD0iMCAwIDUxMiA1MTIiIGZpbGw9IiMwMDAwMDAiPgoKPGcgdHJhbnNmb3JtPSJtYXRyaXgoMi42NSAwIDAgMi42NSAyNTYgMjc2KSI+Cgo8cGF0aCBkPSJNLTMyLTI1Yy0zIDctMjQgMjktMjIgNTEgOCA5MiAzNiAzMCA3OCA1MyAwIDAgNzUtNDIgMTUtMTEwLTE3LTI0LTItNDMtMTMtNTlzLTMwLTE3LTQ0LTIgNiAzNy0xNCA2NyIvPgoKPHBhdGggZD0iTTQyIDIxczktMTgtOC0zMWMxNiAxNyA2IDMyIDYgMzJoLTNDMzYtMTMgMjcgNiAxNC01NiAyOS03MyAwLTg4IDAtNjBoLTljMS0yNC0yMC0xMi04IDUtMSAzNy0yMyA1Mi0yMyA3OC03LTE4IDYtMzIgNi0zMnMtMTggMTUtNyAzNyAzMSAxNyAxNyAyN2MyMiAxNSA1NiA1IDU1LTI3IDEtOCAyMi01IDI0LTNzLTMtNC0xMy00bS01Ni03OGMtNy0yLTUtMTEtMi0xMXM4IDcgMiAxMW0xOSAxYy01LTctMS0xNCA0LTEzczUgMTMtNCAxMyIgZmlsbD0iI2ZmZiIvPgoKPGcgZmlsbD0iI2U5YzEzMiIgc3Ryb2tlPSIjMzMzIiBzdHJva2Utd2lkdGg9IjEiPgoKPHBhdGggZD0iTS00MSAzMWwyMSAzMGMxMSA3IDUgMzUtMjUgMjEtMTctNS0zMS00LTMzLTEzczQtMTAgMy0xNGMtNC0yMiAxNC0xMSAxOS0yMnM1LTE2IDE1LTJNNzEgNDVjLTQtNiAwLTE3LTE0LTE2LTYgMTItMjMgMjQtMjQgMC0xMCAwLTMgMjQtNyAzNS05IDI3IDE3IDI5IDI4IDE2bDI2LTE4YzItMyA1LTYtOS0xN20tOTItOTJjLTMtNiAxMS0xNCAxNi0xNHMxMiA0IDE5IDYgNCA5IDIgMTBTMy0zNS01LTM1cy0xMC04LTE2LTEyIi8+Cgo8cGF0aCBkPSJNLTIxLTQ4YzggNiAxNyAxMSAzNS0zIi8+Cgo8L2c+Cgo8cGF0aCBkPSJNLTEwLTU0Yy0yIDAgMS0yIDItMW03IDFjMS0xLTEtMi0zLTEiLz4KCjwvZz4KCjwvc3ZnPg==&logoColor=black)
![macOS](https://img.shields.io/badge/macOS-555555?style=plastic&logo=apple&logoColor=white)


Spaudible is an offline search engine for discovering new music that's acoustically similar to your favorite songs. It achieves this by converting your input song into a mathematical vector (i.e. a song "fingerprint"), comparing this vector against a quarter billion other song vectors, and returning a playlist of songs that scored the highest in similarity.

<video width="785" height="582" controls muted loop playsinline>
  <source src="https://gist.github.com/Daveofthecave/10eee69210abadf7285202fc9b102c1b/raw/7d5ac72975d82a82f8bb60305a45bd7be178c320/spaudible-search-16fps-AV1.mp4" type='video/mp4; codecs="av01.0.08M.08"'>
</video>

On a modern PC with an Nvidia GPU, finding similar songs takes only a few seconds, thanks in part to the GPU acceleration that PyTorch + CUDA provides. Even on systems without an Nvidia GPU, the highly optimized CPU functions from PyTorch ensure that similarity searches complete within a few minutes.

## How It Works

Every song has its own set of unique attributes; for example:

- Genre (eg. rock, classical, jazz)
- Tempo (eg. 120 bpm)
- Key (eg. C major, F♯ minor)
- Release year (eg. 2007)
- …







.

.

.

.

These similarity calculations are conducted on a custom-built binary cache of 32-dimensional track vectors working in tandem with Spotify's music metadata databases. Each dimension in one of these track vectors encodes a particular attribute of a song; in other words, it represents that song's genre, tempo, key, release year, etc. as a number. If we want to find out how similar two songs are, we compare each pair of these numbers to one another -- genre to genre, tempo to tempo, and so on. If most number pairs are similar, we consider the two songs similar. If most of them are different, then the two songs are different.

























.

.

.

A locally-run song recommendation tool using vector embeddings derived from Spotify's music metadata databases

Spaudible is an offline search engine that helps you discover new music that's acoustically similar to your favorite songs. It achieves this by converting your input song into a mathematical vector (i.e. a song "fingerprint"), comparing this vector against a quarter billion other song vectors, and returning a playlist of songs that yielded the highest similarity scores.
