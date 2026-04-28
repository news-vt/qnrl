# QnRL: Quantum-Native Reinforcement Learning

<!-- Badges -->
[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![JAX](https://img.shields.io/badge/JAX-0.6.0-7E0B8C.svg?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq5MTYAhOxXM%2FmvEpsoWmtfP9RxnkAIAr%2F5pVxqPxH93msKodRSXIct2l0OU0%2FL4eY506L%2B3GyJ6UMEZfjjCDbysNcWWmFweJP0Jz%2FA0g2gk80pGkYAAAAAElFTkSuQmCC)](https://docs.jax.dev)
[![Flax](https://img.shields.io/badge/Flax_NNX->=0.11.2-1FA89A.svg?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAABGdBTUEAALGPC%2FxhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAhGVYSWZNTQAqAAAACAAFARIAAwAAAAEAAQAAARoABQAAAAEAAABKARsABQAAAAEAAABSASgAAwAAAAEAAgAAh2kABAAAAAEAAABaAAAAAAAAAAUAAAABAAAABQAAAAEAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAGaADAAQAAAABAAAAGQAAAABJKSHPAAAACXBIWXMAAADFAAAAxQEdzbqoAAABWWlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iWE1QIENvcmUgNi4wLjAiPgogICA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgogICAgICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyI%2BCiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24%2BCiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY%2BCjwveDp4bXBtZXRhPgoZXuEHAAAGj0lEQVRIx6VWaVBb1xV%2B1E7aSVq3M%2FnTTqdpptNOO22a%2BkemIW0msd3peEjTqVMvCRiDDRhMgDixG681KDYGsYhdICQ2IZBA29P%2BJNAuJAOSsUEIBxsH2yyx3diu42RikN77%2Bp4ncgTutO70%2FLlH95z7fUfnnnfOJYj%2FIi4X1rJLEqcDWOcfvbqXXZ%2F5ypzE6muJ%2F0dYgG%2FE9dFzczv5Iv%2FVTF4ApULfYiB4JTtu4%2FG%2B9nt84A2uh9HPX7%2Bb3CwbjaSe8OI3OSYk5%2Bjur88xYsdxLxq6RmZmZ%2F%2BxIf6vWLq1iYH9WwmFQk8kkK3h1rnrd19qkI4E%2Flpg%2BfLFXWTspVwSL2aQsb%2Fkm%2B8L2gLBj6%2FdeTXRfzXOankYgdl5ocYXvFKQaDSPhUMn2oeQVmRdOi72wRCaiCRGHRq%2Fmkk5pxsS9rj7SopHnBR3PB9efLNaEriQVxWGTHn%2BALd3YWGhiGc3LxN9zdip742VUXbsNihiRJ8Q7D4TmZ8%2FyPmRpqnsvKoJVIoDM2fD8zsSMpJExHOvNIa7C%2FnDILZr8Ua%2BDe7gjLRyiHLUee3R7vPD7Zk21T2OiJDW0kRfC7Js6s%2FUk8FGgXsgWu6jvO6zlzu2FNhAbNUiv2wYCv2EcvVlJ33Y6J0jtpLIPzm4zCddTC6lAUF2YJ9dO4V%2FYpNN5lls0FLYbVXFGkkrbJ3uBXyKP%2BxzkBFC244sSo1ynZMuKB1cJraROFbjvsnifmsFUUWrP%2Fz97Xqc6nZFU4wdbMSNdI6hP6o3DdE2gQOVRB2aX5HQilM6tGxqoyuIWtiqHTCw9n0GZZTobqJf00twWuFa%2FuXbBrBBX2RJvrmChNfojXwv1YE61TBeUIrp3cb%2BWF%2BZnpGly6FpGmB6TuoZya5ulBMCiNOk6PlQz6gabLQ8px%2FyYi1yDMrYj5UiWqAZwq8zXThU7bzMkjy5gqS5J%2FhJvXjENDwzU%2FekpgMsCfqPaaEV2pmGv2tweIsYTX9Tov%2B0Ec0fKHF0qwS1R1QwSJxM7%2FsqZOuVINRtsE9NSkRdZ7X10uHbK9LFtYbZa7c2c7plaixj04AatZ7BeZfUtyyvsOBUngyNR9mc5%2FfgQIoIp9nfTcc1KNnTBUU1BXeXb7nF57yR7NBCOurbz%2BFcYvEWF%2FHUIyXMiefiR6WGybFxdv9nbW9LPxft6EIvm5rKgl5UFPZC0TiA6vf6UMESyng6WpIpgyS9m2H9f2qfmVK4PpqqXV1UceWJmWufbuF03WQwI92mhWTUO0TWU%2Ffr1goh%2BFUTlCd0qNgnA79Qgap35Og7qkVdcgvDFYSmWL9snRyXFlGaL2Uh%2F2EO5%2Fbtz194pLqEstGbNa0jVHD2cu16UzcIhQh7LEp0ySmm4eVWRpzaBeFhJSr396HlsBrCP7Wh%2FtVWyFrNzDGTDs%2FppCDYc%2B6PI5X9mkl%2Bqzy4yJJ8ewVJcb0n8qMsB47UeMHXO%2BmtGlmM6KlnDhlING9sY9qzZbSgSI6TWd2oZYlE2ztp8bZO%2BqCFBOuHjcr2aJl5MFYq8tG%2FL3DjAN9xnSV5egVJWYs%2FnJxrwLM7NdH0ogGU6gdoorc%2BVmoxLasO6SDa3AF5LUWLK82Q11B0w%2FMiaIvNOGLSxVi%2FaIneSu%2F9wI5n3lLH1mfpUVznucaSfGfF5Zxs9M0%2FnabH5ncNS6%2FnmZlykwuEpRtFlJZrhC97yeEFu9ANstgYdQg9cMn9c%2FgCf95jUs4Rxk6cMjnxxxwjk7LfEH0uncShKuct9tx3ia%2FnAEGozZG2d%2Fln8JNMAza9Y8GA%2F2Jn1ZDVUuMdZDpCftUuq%2BoLPmmEtJpkBDozsgc1n%2FWMnek77RlAld9ms%2FmnZSlFFH6YpmG2HQugtTd09uHUTCzhyPTNlNqO4XNFggjk6okHXXh6cXFvmdN6j1CwzVHeGNtgkLFrU4yQC1HutN6fmJsr5Px6yfH9b5WMcT3rHuWaLonfx%2BpP5OFsGPRcLPWHZvMTjdZzkdEDVjN%2BoWpZKrQYYAqFw4l2%2B9Cloz26cYoF%2FcEj30jiRmvrisn4wGHhxq3f1XeNjL2RZ1p6r9wRrVJ7cLDSEX0917wsaD8TvjJ%2FZ2P8kRE%2Fm8vi%2FMcxzIHzeNyM5z1wmr95Z71YHhrJKPFiXaoBP99OLq1L1SPthActstHzc5%2Fc%2FW1iNhLH8P%2F8WglP33izui1wKbs0AHYsXBkbX0hNfK1wwT2SnscVpfJBZGu%2BIn0qELyakVCaax4n8n8Betd4nK8TjTEAAAAASUVORK5CYII%3D)](https://flax.readthedocs.io)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.42.0-FFBB34.svg?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAABGdBTUEAALGPC%2FxhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAhGVYSWZNTQAqAAAACAAFARIAAwAAAAEAAQAAARoABQAAAAEAAABKARsABQAAAAEAAABSASgAAwAAAAEAAgAAh2kABAAAAAEAAABaAAAAAAAAAGAAAAABAAAAYAAAAAEAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAGaADAAQAAAABAAAAGQAAAABElBaaAAAACXBIWXMAAA7EAAAOxAGVKw4bAAABWWlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iWE1QIENvcmUgNi4wLjAiPgogICA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgogICAgICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyI%2BCiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24%2BCiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY%2BCjwveDp4bXBtZXRhPgoZXuEHAAAEyElEQVRIx6WWaWwVZRSG39sWqlJsKfsa2dKWgktQTAQDPzAhxkAMGH9oIv8IokZR1BB%2F4B6j4gKKhOASbdCy9M7MXbpN771tkR1EIcYYNRpJUBBEoNx25vsef9zbciFCBN7kJDOTL%2FOec773LFI9xSrAKlFU%2BE69irXqwm9XhN7%2FV7cyuCrBIvlMlCTSM%2B8mddO8wrOfpLhueJz7ymM8UOmysDzOwsr%2FMpeF5R6LxsZZMLOTgZKkmiamlrkcG54CNXDi900f%2BXQKdlRjUxP2klGNoFRxDta0QkkS1AzFzRD5DytqhqJGKPKhwuPn6S2M041xvhyfAcXpWv5pD9zfCI3TIK2QXbWQ1pn9dclOxUFJTM02Y0Y0YORgih2szpspMCuHLm2HQTHeVJnHlgmtoDjdKz47RY%2Bag3OvrYbtgpaqHr4WrHmJn2q%2BCZ7%2F%2FCRqAsWgKmqMHLojOTNyoM9ckMM5paHC5R2NinNPSTPc6BAqht312GHQZkzDfGy7wI8Yk5gdmLl1nFGK2Irf0FcYJWBSGpTOkaqAKOIQyoVSF6YkuE2SNNBjy8A2kEtWMcvWlUfofnK9pV0YfzKkFYbNszAbnrK8%2B4Q9uvl9NrZ897Ec5t%2FqmWfLXI5fRJRVCoZ6rOlTTm2CKSUuwSAXShwbqhG77JMsfzovnmSnIFMFKWVtp2C7DBlBWmdp1wJJmuqZdZEMRByycggVhwEux%2B5qYliOYT39JGmEx9tqg9KoDaqiJlDunnaT0dO0lQfsHgKto0L8MVh%2FfJZ9U6Gt7Ne6jl8HyeGHsmQ%2BTQ5hkQ%2Bj4yyXJCUo1ZwUJZI0q4NBZQ5%2FKJELe1LUhPJBLgs4qGFBasYeOgV%2BmSGlgI4pkNEvzzQf%2F1jNUOnYUFGMGmGgy%2BHz1dxbyAlKJWlEjEcjPhQ7BHIwQ5Nwhxuc7D1vU7Ud7BDWnxzQLkLv3mBx%2FVnkYYfk7iIoboGxMeZLUm09%2FQubR6T3aYDDATX1EQXyYVGs6wNJCjJzX2dPBdYfG9Ihm312HUs3ZK0SMDxqu%2BVDpUvygnZyAfJp65W0HMIih1Ae3OyGqJFKMlrG3juhQ4HZuohsacw%2Bt%2FE0SmCHR7HFHlQlufnSJJJ6G2GFxzal%2BtTSfVsHyGEN7XqYndXQKRvWLbHdauSVDScCJTg1OgMVHmsvT1AQzaRGaksdjDyQQ%2FctCVCUf3amtm0iJUtqNKZ1KmbeZ%2BbdtT1oH0x27JFRKYbknS25TF8m0ks0JMbqSBsoSs80J7Ty4MP6XwwtMrSNwbbLkKgh8cW%2BJiV5%2BZGGvydK0pzLE%2BQwPV83M1oZXObmJF3bkCvQV947jV27PN%2FXJndzULBbD12c7v%2BHvKTHxXhcPlQ32EBJWLnhtO2a3QBN1ZBWD3vGYNpmrJSkw4fr%2B1%2FZNOO8pEs9vp2YBMUJltadoUu%2B7XnjLewOnWX%2FBMiMfEGSWK9%2BVz4282kbE2fegBbon5fzocUHQPUBsTnQrpCWB2tyfukqR3S%2BJVzvEq30QR7nbt8WWHfVUQ6tafmR7zUttwfUF1%2F1DtAn6QRT5EJ5biD9pQ5QnKZ8BJFrWjTEeUkPdnlVO%2BCGVqhuh7lesKLQkWtDgZcj4yyvcPEUZ8nVyfZ%2FEl1KhZfDv2zF%2B5HMSE9KAAAAAElFTkSuQmCC)](https://pennylane.ai)
[![Gymnasium](https://img.shields.io/badge/Gymnasium->=1.2.1-FFFFFF.svg?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAsCAYAAAAjFjtnAAAACXBIWXMAAAC5AAAAuQHip0HoAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm%2B48GgAABvRJREFUaIGtmXvs1lUZwD%2FvDwH9CaLcBbmZCUn95CIlcvHOcDIvXJqlmaumTs1whkXlH6xNXGnmyDZyK7NpRmOLNqfzGloiQmYtTUXpZ%2FFDUMAh4WpePv3xnDe%2BvXzf73te4NnO9v2e85znPM9zznlup6ZyADASWAScD5wIjAD2AG8CTwEPAQ8CHyb8Xgn3PGA2cCzQH9gKvJJwf53%2B2wO1nXaMeqf6prpCPVsdlsaOUMerV6mPqZvVy9Uvpu9H09j4hIs6XD1H%2FVGieYd6dDs8tcP8FPV19fuZi5ylvqPuUs%2FIVM7t6iZ10qEWYIrao56XiX980vpS9Tvqa%2BrYzLnnp7VOPlQCDFb%2FoZ6bycCwtFOXFfquSJodkkljrtqtDjwUAvxcXZa5cC%2F1CfWmkrGl6iNqRyat76o%2FPVgBxqn%2FVPtmLnq1%2BrBaKxmrGZf7K5m0%2BhpHafTBCLAsaSJnwX7qVvXjFTgTElOdmTRvUW%2BuwuloYWVPBR7PtMhfAh4BNlXgvAw8CVyRSfOJxENzaKGBzYatztHW8%2BqnM%2FCmq89l0hylvnowO%2FAhUMvQ1BhgILAhA%2FdZYDgwKgO3zkNTOKxibAZwOLAS%2BDewE3gBeBTY3IA7E1gL5MQlEuHGDOCBhrHjgTnAJEIhncARwGnAM2XEynZgNvBnYAXwGrAXuA9YB0wm7sRa4NzCnMnAxgzm6%2FDHxGQd5hBCPQacDPwhrbmXUNZdhPJm7Uep4UwtTef%2BnPT%2FKfXvau8CTk29SN1o2PWR6gPqhZnnGnW%2Bep96nGFaNyaaRfPb13BmJ6X%2FOYmXbxRpFYneoP7J%2Fb3f6jRWZte%2FZpjFF9RZbQhwRrr0PYlGmd9Yov6qoW9QWmtxowBTjXBhaAmhkYYzO60JM59M4zPbEOB09Y00t2x8ZuJnRMnY0LTepKIAz6gXt9DYlgohVvv%2Fgd5R6sTEyAzDgRWd1zx1VQXzPS12dJH6lEpNPRW4G%2Bii2orMIqzGTemCFeEnhEGoW4yjgG5gV%2BofRJja7cDTwAeEebyugc4XgOXAJcDvK3ipAS8Cl6Pept6YufUTDCf0oBEej1FXqnvUV9RLjYtZNremnqBeaVzG3epy41yPTTTXqydm8rJEvRV1gzo5cxJGxHljYmCbYbnOUp9tg8ZGI2S%2BTd1uJD3Xmx%2Bpok5T16G%2BZQRiuRM%2Fob6k3uO%2BzKzDCOSGZcwfYdynuuUZbZjj32XOr7cB6lbUvW1MOtO4YAtLxlaqX82gsVj9cUNfLe3q5qSgHF5q6l7UHe5Lsqta3TpMazI%2BWX3Zcptebx1GcNYsXVxgmNcTMvjpr27DcAwntUAek7Z9Rgu8x9VLKsYvNaoTVTQuM45o%2FxZ4XeoG1B%2B22Pqa%2BrRREmmllVOMfLgsYek0jsjUDDp32DqdvEG9HcOCrK9A%2FHKG1opthZFHN%2Fbfm8ZyaBxuHLUqZ7ZBPb3%2Bs97ykknvpLWuNgTom%2BgVfcsSdZ3apw06840CQdnYPCN6%2BF8oMdtwLoMbEBcYDiZ30Xobalzoq9RrjDOdW1Kpt45Eo1F5Q4wodWZRANRvq0%2BqRxb6VqmfOwABMC7%2B24ajGnWANG42Evv6fz%2FDX3yr3ldErhkXep0RDtTS4i2LSyVtnLrWKLHUndS4A6Az1X0eflT6%2FoEFU1026XrDZC4xqmntLDgsKWG3USOqpXat%2Bq56q%2BUhe7PWJ9G60vBB1zXiNJs40bgk7xlVtvEViwww7sr9Rlhyt%2Fq3Erw9Sbi3jGxsgdVF4glG9vV%2B2sVSX1XTphH0mcAy4HlgLpFkbyLeAD4ChhCVhaNTiLwGWJ367gVOaaC3I40dBiwELiRC9F3AFuBt4h1hOPHmsAN4GLgAODuF5%2FtBVVXiPSJmX5z%2BBwIfAwakeHwv0AO80TCvL1HFaIT3gd7Au8DPUgMYSzyYdBL5yG7g9SQYwEVprVKoEqCbKHPUYVeBaBW8QwhbhF7AkcC%2FmqzT3YRWH2KHdzZbrKqwtZ3Q2LEVOGWwhdBosSA2ititj9qk1UUc26bzWlXm1gAL2lz0A0KICYW%2BKUQK2C4sAn5ThdBKgF8AV1N91Mrgt8DFhf%2BFrRgpgX7A54H7K7EybPGaZMfb8QeTkg%2FpY5RlthmVinZoLFfvaoWX61WrSirN2i%2FVryeb%2F802584zYrNBh0IAjGCvx3hWzZ1znLpT%2FYv5LzwYJcoemxe9DkgAjNp%2Ft%2Fo9w%2FtW4Xam7X8pMXOt1akmSdt3GnnAxFy%2BWl3iIjwHTCVs86tEMesCYBxwDDCU8N63EFVtgGnAdGA%2BUV2%2BBphI%2BAWIF%2F65idaLhAOcRhsWqyqUqIIhwGcTw13AYOA%2FwF%2BJito97O%2BhpxNPS58BRqe%2B3YnZh4BVRDjRFvwXSRI37JWYzg4AAAAASUVORK5CYII%3D)](https://gymnasium.farama.org)


<!-- Introduction -->
This repository is the official implementation of the paper "QnRL: Quantum-Native Reinforcement Learning".

## Installation

The codebase is provided as an installable Python package called `qnrl`. To install the package via `pip`, you can run:

```bash
# Navigate to `qnrl` source folder.
$ cd path/to/qnrl/

# Install `qnrl` package.
$ python -m pip install .
```

You can verify the package was successfully install by running:

```bash
$ python -c "import importlib.metadata; version=importlib.metadata.version('qnrl'); print(version)"
1.0.0
```

### Requirements

See [`pyproject.toml`](./pyproject.toml) for package requirements. Package management can be done via `pip` or `uv` (see [`uv.lock`](./uv.lock)).

## CLI

We have developed a lightweight command-line interface (CLI) that can be invoked via the Python interpreter like so:

```bash
# Open the help dialog of the CLI tool.
$ uv run python -m qnrl.cli --help
```

## Package Design Ethos

To simplify interaction with experiment configurations, (nearly) all custom classes within the `qnrl` package inherit from [`mashumaro`](https://github.com/Fatal1ty/mashumaro) to facilitate serialization/deserialization between various formats. This also includes all JAX Flax NNX modules used within the package. As a result, it is very easy to create a configuration in Python using loops, functions, or editor reference tools, and then export to JSON, YAML, or simply Python dictionaries for use elsewhere.

### Algorithm Configurations

In `qnrl` _algorithms_ define the core of experiment implementation. Algorithm configurations are defined as instances of the [`qnrl.algorithms.algorithm.Algorithm`](./qnrl/algorithms/algorithm.py) class, which contain the implementation for the algorithm's training and evaluation procedure. All algorithms can be serialized/deserialized. Variants of this class exist for each algorithm mentioned in the paper, which are:

Algorithm | Class Module Reference
--- | ---
C51 | [`qnrl.algorithms.c51.C51`](./qnrl/algorithms/c51.py)
QnRL | [`qnrl.algorithms.qnrl.QnRL`](./qnrl/algorithms/qnrl.py)

For example, to create a `QnRL` experiment instance on the `CartPole-v1` environment in Python, you can do the following:

```python
from qnrl.optimizer import Partition, AdamW
from qnrl.environment import EnvConfig
from qnrl.modules import QuantumNativeQNetwork, RescaleInput
from qnrl.algorithms.qnrl import QnRL

# Experiment configuration.
config = QnRL(
    seed=0,
    batch_size=128,
    buffer_size=10_000,
    total_timesteps=100_000,
    learning_starts=10_000,
    policy_update_frequency=10,
    target_update_frequency=500,
    gamma=0.99,
    tau=1.0,
    eps_start=1.0,
    eps_end=5.0e-2,
    eps_exploration=0.5,
    reseed_model_on_init=True,
    env=EnvConfig(env_id="CartPole-v1"),
    model=QuantumNativeQNetwork(
        din=4, # Feature dimension of CartPole observations.
        n_actions=2, # Number of actions for CartPole.
        n_returns=2**5, # 32
        z_min=-100,
        z_max=100,
        m_moment=1,
        n_reuploads=1,
        n_layers=3,
        rngs_kwargs=dict(default=0),
        pauli_enc=('RX',),
        pauli_var=('RZ','RY','RZ',),
        imprimitive='offset-CZ',
        input_preprocess=RescaleInput(scale=[2.4, 2.5, 0.21, 2.5]), # Divide by half the upper range to re-scale.
        input_multiplex_activation='tanh',
    ),
    optimizer=Partition(
        transforms={
            'trainable_quantum': AdamW(learning_rate=1e-2, b1=0.7, b2=0.999, weight_decay=5e-3),
            'trainable_classical': AdamW(learning_rate=1e-3, b1=0.7, b2=0.999, weight_decay=5e-3),
        },
        pytree={
            'circ_weights_var': 'trainable_quantum',
            'circ_weights_enc_obs_kernel': 'trainable_quantum',
            'linear_input_projection': {
                'bias': 'trainable_classical',
                'kernel': 'trainable_classical',
            }
        },
    ),
)
```

And then dump to JSON:

```python
# Dump the configuration as JSON.
import json
print(json.dumps(config.to_dict(), indent=4))
```
```json
{
  "type": "QnRL",
  "env": {
    "type": "EnvConfig",
    "env_id": "CartPole-v1",
    "render_mode": null,
    "max_episode_steps": null,
    "extra_args": null
  },
  "seed": 0,
  "metric_manager": null,
  "checkpoint": null,
  "train_meta_prefix_key": "train/",
  "eval_meta_prefix_key": "eval/",
  "model": {
    "type": "QuantumNativeQNetwork",
    "din": 4,
    "n_actions": 2,
    "n_returns": 32,
    "z_min": -100,
    "z_max": 100,
    "m_moment": 1,
    "n_reuploads": 1,
    "n_layers": 3,
    "rngs_kwargs": {
      "default": 0
    },
    "pauli_enc": [
      "RX"
    ],
    "pauli_var": [
      "RZ",
      "RY",
      "RZ"
    ],
    "imprimitive": "offset-CZ",
    "use_qft": true,
    "input_preprocess": {
      "type": "RescaleInput",
      "scale": [
        2.4,
        2.5,
        0.21,
        2.5
      ]
    },
    "input_multiplex_activation": "tanh"
  },
  "optimizer": {
    "type": "Partition",
    "transforms": {
      "trainable_quantum": {
        "type": "AdamW",
        "learning_rate": 0.01,
        "b1": 0.7,
        "b2": 0.999,
        "eps": 1e-08,
        "eps_root": 0.0,
        "mu_dtype": null,
        "weight_decay": 0.005,
        "nesterov": false
      },
      "trainable_classical": {
        "type": "AdamW",
        "learning_rate": 0.001,
        "b1": 0.7,
        "b2": 0.999,
        "eps": 1e-08,
        "eps_root": 0.0,
        "mu_dtype": null,
        "weight_decay": 0.005,
        "nesterov": false
      }
    },
    "pytree": {
      "circ_weights_var": "trainable_quantum",
      "circ_weights_enc_obs_kernel": "trainable_quantum",
      "linear_input_projection": {
        "bias": "trainable_classical",
        "kernel": "trainable_classical"
      }
    }
  },
  "buffer_size": 10000,
  "batch_size": 128,
  "total_timesteps": 100000,
  "learning_starts": 10000,
  "policy_update_frequency": 10,
  "target_update_frequency": 500,
  "gamma": 0.99,
  "tau": 1.0,
  "eps_start": 1.0,
  "eps_end": 0.05,
  "eps_exploration": 0.5,
  "reseed_model_on_init": true,
  "eps_eval": 0.0
}
```

Or dump as YAML:


```python
import yaml
print(yaml.dump(config.to_dict()))
```
```yaml
batch_size: 128
buffer_size: 10000
checkpoint: null
env:
  env_id: CartPole-v1
  extra_args: null
  max_episode_steps: null
  render_mode: null
  type: EnvConfig
eps_end: 0.05
eps_eval: 0.0
eps_exploration: 0.5
eps_start: 1.0
eval_meta_prefix_key: eval/
gamma: 0.99
learning_starts: 10000
metric_manager: null
model:
  din: 4
  imprimitive: offset-CZ
  input_multiplex_activation: tanh
  input_preprocess:
    scale:
    - 2.4
    - 2.5
    - 0.21
    - 2.5
    type: RescaleInput
  m_moment: 1
  n_actions: 2
  n_layers: 3
  n_returns: 32
  n_reuploads: 1
  pauli_enc:
  - RX
  pauli_var:
  - RZ
  - RY
  - RZ
  rngs_kwargs:
    default: 0
  type: QuantumNativeQNetwork
  use_qft: true
  z_max: 100
  z_min: -100
optimizer:
  pytree:
    circ_weights_enc_obs_kernel: trainable_quantum
    circ_weights_var: trainable_quantum
    linear_input_projection:
      bias: trainable_classical
      kernel: trainable_classical
  transforms:
    trainable_classical:
      b1: 0.7
      b2: 0.999
      eps: 1.0e-08
      eps_root: 0.0
      learning_rate: 0.001
      mu_dtype: null
      nesterov: false
      type: AdamW
      weight_decay: 0.005
    trainable_quantum:
      b1: 0.7
      b2: 0.999
      eps: 1.0e-08
      eps_root: 0.0
      learning_rate: 0.01
      mu_dtype: null
      nesterov: false
      type: AdamW
      weight_decay: 0.005
  type: Partition
policy_update_frequency: 10
reseed_model_on_init: true
seed: 0
target_update_frequency: 500
tau: 1.0
total_timesteps: 100000
train_meta_prefix_key: train/
type: QnRL
```


## Training 

We use [WandB](https://wandb.ai) to manage hyperparameter sweeps and track experiment results.

> [!IMPORTANT]
> For all training you should create an `.env` file with key `WANDB_API_KEY`.

### Experiment Configurations

The experiment configuration for each of the frameworks and environments discussed in the paper is described as a YAML file in the [`./sweeps`](./sweeps) folder.

The full list of experiments is as follows:

Experiment YAML File | Description
--- | ---
[`sweeps/c51_Acrobot-v1.yaml`](./sweeps/c51_Acrobot-v1.yaml) | C51 algorithm on the Acrobot environment.
[`sweeps/qnrl_Acrobot-v1.yaml`](./sweeps/qnrl_Acrobot-v1.yaml) | QnRL algorithm on the Acrobot environment.
[`sweeps/c51_CartPole-v1.yaml`](./sweeps/c51_CartPole-v1.yaml) | C51 algorithm on the CartPole environment.
[`sweeps/qnrl_CartPole-v1.yaml`](./sweeps/qnrl_CartPole-v1.yaml) | QnRL algorithm on the CartPole environment.
[`sweeps/c51_FrozenLake-v1.yaml`](./sweeps/c51_FrozenLake-v1.yaml) | C51 algorithm on the FrozenLake environment.
[`sweeps/qnrl_FrozenLake-v1.yaml`](./sweeps/qnrl_FrozenLake-v1.yaml) | QnRL algorithm on the FrozenLake environment.
[`sweeps/c51_CliffWalking-v1.yaml`](./sweeps/c51_CliffWalking-v1.yaml) | C51 algorithm on the CliffWalking environment.
[`sweeps/qnrl_CliffWalking-v1.yaml`](./sweeps/qnrl_CliffWalking-v1.yaml) | QnRL algorithm on the CliffWalking environment.
[`sweeps/c51_Breakout-v5.yaml`](./sweeps/c51_Breakout-v5.yaml) | C51 algorithm on the Breakout environment.
[`sweeps/qnrl_Breakout-v5.yaml`](./sweeps/qnrl_Breakout-v5.yaml) | QnRL algorithm on the Breakout environment.
[`sweeps/c51_SpaceInvaders-v5.yaml`](./sweeps/c51_SpaceInvaders-v5.yaml) | C51 algorithm on the SpaceInvaders environment.
[`sweeps/qnrl_SpaceInvaders-v5.yaml`](./sweeps/qnrl_SpaceInvaders-v5.yaml) | QnRL algorithm on the SpaceInvaders environment.

### Sweeps using WandB

[WandB](https://wandb.ai) makes it very easy to run hyperparameter sweeps to compare the performance of multiple configurations. The following sections detail how you can run our preconfigured sweeps using the `wandb` API.

#### Local computer

On your personal computer, start a new sweep:

```bash
# Start a sweep using YAML config via UV.
$ uv run --env-file .env wandb sweep sweeps/qnrl_CartPole-v1.yaml
wandb: Creating sweep from: sweeps/qnrl_CartPole-v1.yaml
wandb: Creating sweep with ID: 4ykq4hvw
wandb: View sweep at: https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/4ykq4hvw
wandb: Run sweep agent with: wandb agent zanderman-virginia-tech/qnrl/4ykq4hvw
# Copy the agent ID string: "zanderman-virginia-tech/qnrl/4ykq4hvw"
# Paste the agent ID string into the following command:
# Start 1 agent that performs 1 run of the sweep.
$ uv run --env-file .env wandb agent --count 1 zanderman-virginia-tech/qnrl/4ykq4hvw
```

#### SLURM

Source `.env`:

```bash
# Load Anaconda from SLURM modules.
$ module load Miniconda3
# Load Python environment from Anaconda.
$ conda activate qnrl
# Source environment file, which should have WandB API key.
$ source .env
# Start a sweep using YAML config file and pass WandB API key from environment.
$ WANDB_API_KEY=$WANDB_API_KEY wandb sweep sweeps/qnrl_CartPole-v1.yaml
wandb: Creating sweep from: sweeps/qnrl_CartPole-v1.yaml
wandb: Creating sweep with ID: 4ykq4hvw
wandb: View sweep at: https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/4ykq4hvw
wandb: Run sweep agent with: wandb agent zanderman-virginia-tech/qnrl/4ykq4hvw
# Copy the agent ID string: "zanderman-virginia-tech/qnrl/4ykq4hvw"
# Paste the agent ID string into the following command:
# Start 100 agents, max of 10 running at a time, that each perform 1 run of the sweep.
$ sbatch --array=1-100%10 scripts/slurm-wandb.sh agent --count 1 zanderman-virginia-tech/qnrl/4ykq4hvw
```

### Single-run Training

You can also train single hyperparameter configuration instances using our CLI via the command `python -m qnrl.cli train`. This commands takes a `--config="{...}"` argument that is **a Python dictionary formatted as a string**. The training CLI requires this type of configuration formatting to be seamlessly compatible with the WandB sweep API. As such, these are the same training commands that a WandB agent runs under the hood.

```bash
# Train a configuration of QnRL for the CartPole environment.
$ uv run --env-file .env python -m qnrl.cli train --config="{'batch_size': 128, 'buffer_size': 10000, 'env': {'env_id': 'CartPole-v1', 'type': 'EnvConfig'}, 'eps_end': 0.05, 'eps_exploration': 0.5, 'eps_start': 1, 'gamma': 0.99, 'learning_starts': 10000, 'model': {'din': 4, 'imprimitive': 'offset-CZ', 'input_multiplex_activation': 'tanh', 'input_preprocess': {'scale': [2.4, 2.5, 0.21, 2.5], 'type': 'RescaleInput'}, 'm_moment': 1, 'n_actions': 2, 'n_layers': 7, 'n_returns': 32, 'n_reuploads': 1, 'pauli_enc': ['RX'], 'pauli_var': ['RZ', 'RY', 'RZ'], 'rngs_kwargs': {'default': 0}, 'type': 'QuantumNativeQNetwork', 'z_max': 100, 'z_min': -100}, 'num_atoms': 32, 'optimizer': {'pytree': {'circ_weights_enc_obs_kernel': 'trainable_quantum', 'circ_weights_var': 'trainable_quantum', 'linear_input_projection': {'bias': 'trainable_classical', 'kernel': 'trainable_classical'}}, 'transforms': {'trainable_classical': {'b1': 0.7, 'b2': 0.999, 'learning_rate': 0.001, 'type': 'AdamW', 'weight_decay': 0.005}, 'trainable_quantum': {'b1': 0.7, 'b2': 0.999, 'learning_rate': 0.01, 'type': 'AdamW', 'weight_decay': 0.005}}, 'type': 'Partition'}, 'policy_update_frequency': 10, 'reseed_model_on_init': True, 'seed': 21, 'target_update_frequency': 500, 'tau': 1, 'total_timesteps': 100000, 'type': 'QnRL'}" --eval_seed=1024 --num_eval=10
```

## Experiment Results

### WandB Sweeps

All our experiment results are tracked using WandB and made public online, which can be found at the URL <https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps>.

The results for each of our experiment sweeps are as follows:

Algorithm | Environment | WandB Sweep ID and URL | Sweep YAML File
--- | --- | --- | ---
C51 | `Acrobot-v1` | [`s3xwn85k`](https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/s3xwn85k) | [`sweeps/c51_Acrobot-v1.yaml`](./sweeps/c51_Acrobot-v1.yaml)
QnRL | `Acrobot-v1` | [`sd4va92r`](https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/sd4va92r) | [`sweeps/qnrl_Acrobot-v1.yaml`](./sweeps/qnrl_Acrobot-v1.yaml)
C51 | `CartPole-v1` | [`7raha9om`](https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/7raha9om) | [`sweeps/c51_CartPole-v1.yaml`](./sweeps/c51_CartPole-v1.yaml)
QnRL | `CartPole-v1` | [`4ykq4hvw`](https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/4ykq4hvw) | [`sweeps/qnrl_CartPole-v1.yaml`](./sweeps/qnrl_CartPole-v1.yaml)
C51 | `FrozenLake-v1` | [`3exa9jvo`](https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/3exa9jvo) | [`sweeps/c51_FrozenLake-v1.yaml`](./sweeps/c51_FrozenLake-v1.yaml)
QnRL | `FrozenLake-v1` | [`sgi6zb0t`](https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/sgi6zb0t) | [`sweeps/qnrl_FrozenLake-v1.yaml`](./sweeps/qnrl_FrozenLake-v1.yaml)
C51 | `CliffWalking-v1` | [`1nlfftsy`](https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/1nlfftsy) | [`sweeps/c51_CliffWalking-v1.yaml`](./sweeps/c51_CliffWalking-v1.yaml)
QnRL | `CliffWalking-v1` | [`h4g5er22`](https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/h4g5er22) | [`sweeps/qnrl_CliffWalking-v1.yaml`](./sweeps/qnrl_CliffWalking-v1.yaml)
C51 | `Breakout-v5` | [`gxpmvyxv`](https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/gxpmvyxv) | [`sweeps/c51_Breakout-v5.yaml`](./sweeps/c51_Breakout-v5.yaml)
QnRL | `Breakout-v5` | [`yavb5d6j`](https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/yavb5d6j) | [`sweeps/qnrl_Breakout-v5.yaml`](./sweeps/qnrl_Breakout-v5.yaml)
C51 | `SpaceInvaders-v5` | [`zo5jkyjm`](https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/zo5jkyjm) | [`sweeps/c51_SpaceInvaders-v5.yaml`](./sweeps/c51_SpaceInvaders-v5.yaml)
QnRL | `SpaceInvaders-v5` | [`x8921x7e`](https://wandb.ai/zanderman-virginia-tech/qnrl/sweeps/x8921x7e) | [`sweeps/qnrl_SpaceInvaders-v5.yaml`](./sweeps/qnrl_SpaceInvaders-v5.yaml)



### Figures

> [!IMPORTANT]
> **TL;DR**: Steps to produce figures:
> 1. Download experiment data using [`notebooks/wandb-download.ipynb`](./notebooks/wandb-download.ipynb) (modify as needed).
> 2. Generate figures using [`notebooks/wandb-vis.ipynb`](./notebooks/wandb-vis.ipynb) (modify as needed).

All figures are generated from the WandB experiment results using the Jupyter notebook tool [`notebooks/wandb-vis.ipynb`](./notebooks/wandb-vis.ipynb). This tool notebook requires that the experiment results be downloaded onto your system prior to work properly. You can use the dedicated download tool notebook [`notebooks/wandb-download.ipynb`](./notebooks/wandb-download.ipynb) to download the data for all the sweeps and runs desired. Both the download and visualization tool notebooks can be modified as needed to produce the desired figures.

The figures that aggregate the metric performance for each of the experiments are given in the table below:

> [!NOTE]
> All figures have both a `*.png` and `*.pdf` variant.

Figure | File | Description
--- | --- | ---
[![figure](./figures/env=Acrobot-v1_type=ablation_model=c51_metric=eval-episode-reward.png)](./figures/env=Acrobot-v1_type=ablation_model=c51_metric=eval-episode-reward.png) | `env=Acrobot-v1_type=ablation_model=c51_metric=eval-episode-reward` [[png](./figures/env=Acrobot-v1_type=ablation_model=c51_metric=eval-episode-reward.png), [pdf](./figures/env=Acrobot-v1_type=ablation_model=c51_metric=eval-episode-reward.pdf)] | Ablation study for C51 on Acrobot showing metric `eval-episode-reward`
[![figure](./figures/env=Acrobot-v1_type=ablation_model=c51_metric=total-model-parameter-count.png)](./figures/env=Acrobot-v1_type=ablation_model=c51_metric=total-model-parameter-count.png) | `env=Acrobot-v1_type=ablation_model=c51_metric=total-model-parameter-count` [[png](./figures/env=Acrobot-v1_type=ablation_model=c51_metric=total-model-parameter-count.png), [pdf](./figures/env=Acrobot-v1_type=ablation_model=c51_metric=total-model-parameter-count.pdf)] | Ablation study for C51 on Acrobot showing metric `total-model-parameter-count`
[![figure](./figures/env=Acrobot-v1_type=ablation_model=qnrl_metric=eval-episode-reward.png)](./figures/env=Acrobot-v1_type=ablation_model=qnrl_metric=eval-episode-reward.png) | `env=Acrobot-v1_type=ablation_model=qnrl_metric=eval-episode-reward` [[png](./figures/env=Acrobot-v1_type=ablation_model=qnrl_metric=eval-episode-reward.png), [pdf](./figures/env=Acrobot-v1_type=ablation_model=qnrl_metric=eval-episode-reward.pdf)] | Ablation study for QnRL on Acrobot showing metric `eval-episode-reward`
[![figure](./figures/env=Acrobot-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.png)](./figures/env=Acrobot-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.png) | `env=Acrobot-v1_type=ablation_model=qnrl_metric=total-model-parameter-count` [[png](./figures/env=Acrobot-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.png), [pdf](./figures/env=Acrobot-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.pdf)] | Ablation study for QnRL on Acrobot showing metric `total-model-parameter-count`
[![figure](./figures/env=Acrobot-v1_type=compare-best_metric=eval-episode-reward.png)](./figures/env=Acrobot-v1_type=compare-best_metric=eval-episode-reward.png) | `env=Acrobot-v1_type=compare-best_metric=eval-episode-reward` [[png](./figures/env=Acrobot-v1_type=compare-best_metric=eval-episode-reward.png), [pdf](./figures/env=Acrobot-v1_type=compare-best_metric=eval-episode-reward.pdf)] | Comparison of best models on Acrobot showing metric `eval-episode-reward`
[![figure](./figures/env=Acrobot-v1_type=compare-best_metric=total-model-parameter-count.png)](./figures/env=Acrobot-v1_type=compare-best_metric=total-model-parameter-count.png) | `env=Acrobot-v1_type=compare-best_metric=total-model-parameter-count` [[png](./figures/env=Acrobot-v1_type=compare-best_metric=total-model-parameter-count.png), [pdf](./figures/env=Acrobot-v1_type=compare-best_metric=total-model-parameter-count.pdf)] | Comparison of best models on Acrobot showing metric `total-model-parameter-count`
[![figure](./figures/env=Acrobot-v1_type=compare-best_metric=train-episode-reward.png)](./figures/env=Acrobot-v1_type=compare-best_metric=train-episode-reward.png) | `env=Acrobot-v1_type=compare-best_metric=train-episode-reward` [[png](./figures/env=Acrobot-v1_type=compare-best_metric=train-episode-reward.png), [pdf](./figures/env=Acrobot-v1_type=compare-best_metric=train-episode-reward.pdf)] | Comparison of best models on Acrobot showing metric `train-episode-reward`
[![figure](./figures/env=Breakout-v5_type=ablation_model=c51_metric=eval-episode-reward.png)](./figures/env=Breakout-v5_type=ablation_model=c51_metric=eval-episode-reward.png) | `env=Breakout-v5_type=ablation_model=c51_metric=eval-episode-reward` [[png](./figures/env=Breakout-v5_type=ablation_model=c51_metric=eval-episode-reward.png), [pdf](./figures/env=Breakout-v5_type=ablation_model=c51_metric=eval-episode-reward.pdf)] | Ablation study for C51 on Breakout showing metric `eval-episode-reward`
[![figure](./figures/env=Breakout-v5_type=ablation_model=c51_metric=total-model-parameter-count.png)](./figures/env=Breakout-v5_type=ablation_model=c51_metric=total-model-parameter-count.png) | `env=Breakout-v5_type=ablation_model=c51_metric=total-model-parameter-count` [[png](./figures/env=Breakout-v5_type=ablation_model=c51_metric=total-model-parameter-count.png), [pdf](./figures/env=Breakout-v5_type=ablation_model=c51_metric=total-model-parameter-count.pdf)] | Ablation study for C51 on Breakout showing metric `total-model-parameter-count`
[![figure](./figures/env=Breakout-v5_type=ablation_model=qnrl_metric=eval-episode-reward.png)](./figures/env=Breakout-v5_type=ablation_model=qnrl_metric=eval-episode-reward.png) | `env=Breakout-v5_type=ablation_model=qnrl_metric=eval-episode-reward` [[png](./figures/env=Breakout-v5_type=ablation_model=qnrl_metric=eval-episode-reward.png), [pdf](./figures/env=Breakout-v5_type=ablation_model=qnrl_metric=eval-episode-reward.pdf)] | Ablation study for QnRL on Breakout showing metric `eval-episode-reward`
[![figure](./figures/env=Breakout-v5_type=ablation_model=qnrl_metric=total-model-parameter-count.png)](./figures/env=Breakout-v5_type=ablation_model=qnrl_metric=total-model-parameter-count.png) | `env=Breakout-v5_type=ablation_model=qnrl_metric=total-model-parameter-count` [[png](./figures/env=Breakout-v5_type=ablation_model=qnrl_metric=total-model-parameter-count.png), [pdf](./figures/env=Breakout-v5_type=ablation_model=qnrl_metric=total-model-parameter-count.pdf)] | Ablation study for QnRL on Breakout showing metric `total-model-parameter-count`
[![figure](./figures/env=Breakout-v5_type=compare-best_metric=eval-episode-reward.png)](./figures/env=Breakout-v5_type=compare-best_metric=eval-episode-reward.png) | `env=Breakout-v5_type=compare-best_metric=eval-episode-reward` [[png](./figures/env=Breakout-v5_type=compare-best_metric=eval-episode-reward.png), [pdf](./figures/env=Breakout-v5_type=compare-best_metric=eval-episode-reward.pdf)] | Comparison of best models on Breakout showing metric `eval-episode-reward`
[![figure](./figures/env=Breakout-v5_type=compare-best_metric=total-model-parameter-count.png)](./figures/env=Breakout-v5_type=compare-best_metric=total-model-parameter-count.png) | `env=Breakout-v5_type=compare-best_metric=total-model-parameter-count` [[png](./figures/env=Breakout-v5_type=compare-best_metric=total-model-parameter-count.png), [pdf](./figures/env=Breakout-v5_type=compare-best_metric=total-model-parameter-count.pdf)] | Comparison of best models on Breakout showing metric `total-model-parameter-count`
[![figure](./figures/env=Breakout-v5_type=compare-best_metric=train-episode-reward.png)](./figures/env=Breakout-v5_type=compare-best_metric=train-episode-reward.png) | `env=Breakout-v5_type=compare-best_metric=train-episode-reward` [[png](./figures/env=Breakout-v5_type=compare-best_metric=train-episode-reward.png), [pdf](./figures/env=Breakout-v5_type=compare-best_metric=train-episode-reward.pdf)] | Comparison of best models on Breakout showing metric `train-episode-reward`
[![figure](./figures/env=CartPole-v1_type=ablation_model=c51_metric=eval-episode-reward.png)](./figures/env=CartPole-v1_type=ablation_model=c51_metric=eval-episode-reward.png) | `env=CartPole-v1_type=ablation_model=c51_metric=eval-episode-reward` [[png](./figures/env=CartPole-v1_type=ablation_model=c51_metric=eval-episode-reward.png), [pdf](./figures/env=CartPole-v1_type=ablation_model=c51_metric=eval-episode-reward.pdf)] | Ablation study for C51 on CartPole showing metric `eval-episode-reward`
[![figure](./figures/env=CartPole-v1_type=ablation_model=c51_metric=total-model-parameter-count.png)](./figures/env=CartPole-v1_type=ablation_model=c51_metric=total-model-parameter-count.png) | `env=CartPole-v1_type=ablation_model=c51_metric=total-model-parameter-count` [[png](./figures/env=CartPole-v1_type=ablation_model=c51_metric=total-model-parameter-count.png), [pdf](./figures/env=CartPole-v1_type=ablation_model=c51_metric=total-model-parameter-count.pdf)] | Ablation study for C51 on CartPole showing metric `total-model-parameter-count`
[![figure](./figures/env=CartPole-v1_type=ablation_model=qnrl_metric=eval-episode-reward.png)](./figures/env=CartPole-v1_type=ablation_model=qnrl_metric=eval-episode-reward.png) | `env=CartPole-v1_type=ablation_model=qnrl_metric=eval-episode-reward` [[png](./figures/env=CartPole-v1_type=ablation_model=qnrl_metric=eval-episode-reward.png), [pdf](./figures/env=CartPole-v1_type=ablation_model=qnrl_metric=eval-episode-reward.pdf)] | Ablation study for QnRL on CartPole showing metric `eval-episode-reward`
[![figure](./figures/env=CartPole-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.png)](./figures/env=CartPole-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.png) | `env=CartPole-v1_type=ablation_model=qnrl_metric=total-model-parameter-count` [[png](./figures/env=CartPole-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.png), [pdf](./figures/env=CartPole-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.pdf)] | Ablation study for QnRL on CartPole showing metric `total-model-parameter-count`
[![figure](./figures/env=CartPole-v1_type=compare-best_metric=eval-episode-reward.png)](./figures/env=CartPole-v1_type=compare-best_metric=eval-episode-reward.png) | `env=CartPole-v1_type=compare-best_metric=eval-episode-reward` [[png](./figures/env=CartPole-v1_type=compare-best_metric=eval-episode-reward.png), [pdf](./figures/env=CartPole-v1_type=compare-best_metric=eval-episode-reward.pdf)] | Comparison of best models on CartPole showing metric `eval-episode-reward`
[![figure](./figures/env=CartPole-v1_type=compare-best_metric=total-model-parameter-count.png)](./figures/env=CartPole-v1_type=compare-best_metric=total-model-parameter-count.png) | `env=CartPole-v1_type=compare-best_metric=total-model-parameter-count` [[png](./figures/env=CartPole-v1_type=compare-best_metric=total-model-parameter-count.png), [pdf](./figures/env=CartPole-v1_type=compare-best_metric=total-model-parameter-count.pdf)] | Comparison of best models on CartPole showing metric `total-model-parameter-count`
[![figure](./figures/env=CartPole-v1_type=compare-best_metric=train-episode-reward.png)](./figures/env=CartPole-v1_type=compare-best_metric=train-episode-reward.png) | `env=CartPole-v1_type=compare-best_metric=train-episode-reward` [[png](./figures/env=CartPole-v1_type=compare-best_metric=train-episode-reward.png), [pdf](./figures/env=CartPole-v1_type=compare-best_metric=train-episode-reward.pdf)] | Comparison of best models on CartPole showing metric `train-episode-reward`
[![figure](./figures/env=CliffWalking-v1_type=ablation_model=c51_metric=eval-episode-reward.png)](./figures/env=CliffWalking-v1_type=ablation_model=c51_metric=eval-episode-reward.png) | `env=CliffWalking-v1_type=ablation_model=c51_metric=eval-episode-reward` [[png](./figures/env=CliffWalking-v1_type=ablation_model=c51_metric=eval-episode-reward.png), [pdf](./figures/env=CliffWalking-v1_type=ablation_model=c51_metric=eval-episode-reward.pdf)] | Ablation study for C51 on CliffWalking showing metric `eval-episode-reward`
[![figure](./figures/env=CliffWalking-v1_type=ablation_model=c51_metric=total-model-parameter-count.png)](./figures/env=CliffWalking-v1_type=ablation_model=c51_metric=total-model-parameter-count.png) | `env=CliffWalking-v1_type=ablation_model=c51_metric=total-model-parameter-count` [[png](./figures/env=CliffWalking-v1_type=ablation_model=c51_metric=total-model-parameter-count.png), [pdf](./figures/env=CliffWalking-v1_type=ablation_model=c51_metric=total-model-parameter-count.pdf)] | Ablation study for C51 on CliffWalking showing metric `total-model-parameter-count`
[![figure](./figures/env=CliffWalking-v1_type=ablation_model=qnrl_metric=eval-episode-reward.png)](./figures/env=CliffWalking-v1_type=ablation_model=qnrl_metric=eval-episode-reward.png) | `env=CliffWalking-v1_type=ablation_model=qnrl_metric=eval-episode-reward` [[png](./figures/env=CliffWalking-v1_type=ablation_model=qnrl_metric=eval-episode-reward.png), [pdf](./figures/env=CliffWalking-v1_type=ablation_model=qnrl_metric=eval-episode-reward.pdf)] | Ablation study for QnRL on CliffWalking showing metric `eval-episode-reward`
[![figure](./figures/env=CliffWalking-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.png)](./figures/env=CliffWalking-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.png) | `env=CliffWalking-v1_type=ablation_model=qnrl_metric=total-model-parameter-count` [[png](./figures/env=CliffWalking-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.png), [pdf](./figures/env=CliffWalking-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.pdf)] | Ablation study for QnRL on CliffWalking showing metric `total-model-parameter-count`
[![figure](./figures/env=CliffWalking-v1_type=compare-best_metric=eval-episode-reward.png)](./figures/env=CliffWalking-v1_type=compare-best_metric=eval-episode-reward.png) | `env=CliffWalking-v1_type=compare-best_metric=eval-episode-reward` [[png](./figures/env=CliffWalking-v1_type=compare-best_metric=eval-episode-reward.png), [pdf](./figures/env=CliffWalking-v1_type=compare-best_metric=eval-episode-reward.pdf)] | Comparison of best models on CliffWalking showing metric `eval-episode-reward`
[![figure](./figures/env=CliffWalking-v1_type=compare-best_metric=total-model-parameter-count.png)](./figures/env=CliffWalking-v1_type=compare-best_metric=total-model-parameter-count.png) | `env=CliffWalking-v1_type=compare-best_metric=total-model-parameter-count` [[png](./figures/env=CliffWalking-v1_type=compare-best_metric=total-model-parameter-count.png), [pdf](./figures/env=CliffWalking-v1_type=compare-best_metric=total-model-parameter-count.pdf)] | Comparison of best models on CliffWalking showing metric `total-model-parameter-count`
[![figure](./figures/env=CliffWalking-v1_type=compare-best_metric=train-episode-reward.png)](./figures/env=CliffWalking-v1_type=compare-best_metric=train-episode-reward.png) | `env=CliffWalking-v1_type=compare-best_metric=train-episode-reward` [[png](./figures/env=CliffWalking-v1_type=compare-best_metric=train-episode-reward.png), [pdf](./figures/env=CliffWalking-v1_type=compare-best_metric=train-episode-reward.pdf)] | Comparison of best models on CliffWalking showing metric `train-episode-reward`
[![figure](./figures/env=FrozenLake-v1_type=ablation_model=c51_metric=eval-episode-reward.png)](./figures/env=FrozenLake-v1_type=ablation_model=c51_metric=eval-episode-reward.png) | `env=FrozenLake-v1_type=ablation_model=c51_metric=eval-episode-reward` [[png](./figures/env=FrozenLake-v1_type=ablation_model=c51_metric=eval-episode-reward.png), [pdf](./figures/env=FrozenLake-v1_type=ablation_model=c51_metric=eval-episode-reward.pdf)] | Ablation study for C51 on FrozenLake showing metric `eval-episode-reward`
[![figure](./figures/env=FrozenLake-v1_type=ablation_model=c51_metric=total-model-parameter-count.png)](./figures/env=FrozenLake-v1_type=ablation_model=c51_metric=total-model-parameter-count.png) | `env=FrozenLake-v1_type=ablation_model=c51_metric=total-model-parameter-count` [[png](./figures/env=FrozenLake-v1_type=ablation_model=c51_metric=total-model-parameter-count.png), [pdf](./figures/env=FrozenLake-v1_type=ablation_model=c51_metric=total-model-parameter-count.pdf)] | Ablation study for C51 on FrozenLake showing metric `total-model-parameter-count`
[![figure](./figures/env=FrozenLake-v1_type=ablation_model=qnrl_metric=eval-episode-reward.png)](./figures/env=FrozenLake-v1_type=ablation_model=qnrl_metric=eval-episode-reward.png) | `env=FrozenLake-v1_type=ablation_model=qnrl_metric=eval-episode-reward` [[png](./figures/env=FrozenLake-v1_type=ablation_model=qnrl_metric=eval-episode-reward.png), [pdf](./figures/env=FrozenLake-v1_type=ablation_model=qnrl_metric=eval-episode-reward.pdf)] | Ablation study for QnRL on FrozenLake showing metric `eval-episode-reward`
[![figure](./figures/env=FrozenLake-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.png)](./figures/env=FrozenLake-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.png) | `env=FrozenLake-v1_type=ablation_model=qnrl_metric=total-model-parameter-count` [[png](./figures/env=FrozenLake-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.png), [pdf](./figures/env=FrozenLake-v1_type=ablation_model=qnrl_metric=total-model-parameter-count.pdf)] | Ablation study for QnRL on FrozenLake showing metric `total-model-parameter-count`
[![figure](./figures/env=FrozenLake-v1_type=compare-best_metric=eval-episode-reward.png)](./figures/env=FrozenLake-v1_type=compare-best_metric=eval-episode-reward.png) | `env=FrozenLake-v1_type=compare-best_metric=eval-episode-reward` [[png](./figures/env=FrozenLake-v1_type=compare-best_metric=eval-episode-reward.png), [pdf](./figures/env=FrozenLake-v1_type=compare-best_metric=eval-episode-reward.pdf)] | Comparison of best models on FrozenLake showing metric `eval-episode-reward`
[![figure](./figures/env=FrozenLake-v1_type=compare-best_metric=total-model-parameter-count.png)](./figures/env=FrozenLake-v1_type=compare-best_metric=total-model-parameter-count.png) | `env=FrozenLake-v1_type=compare-best_metric=total-model-parameter-count` [[png](./figures/env=FrozenLake-v1_type=compare-best_metric=total-model-parameter-count.png), [pdf](./figures/env=FrozenLake-v1_type=compare-best_metric=total-model-parameter-count.pdf)] | Comparison of best models on FrozenLake showing metric `total-model-parameter-count`
[![figure](./figures/env=FrozenLake-v1_type=compare-best_metric=train-episode-reward.png)](./figures/env=FrozenLake-v1_type=compare-best_metric=train-episode-reward.png) | `env=FrozenLake-v1_type=compare-best_metric=train-episode-reward` [[png](./figures/env=FrozenLake-v1_type=compare-best_metric=train-episode-reward.png), [pdf](./figures/env=FrozenLake-v1_type=compare-best_metric=train-episode-reward.pdf)] | Comparison of best models on FrozenLake showing metric `train-episode-reward`
[![figure](./figures/env=SpaceInvaders-v5_type=ablation_model=c51_metric=eval-episode-reward.png)](./figures/env=SpaceInvaders-v5_type=ablation_model=c51_metric=eval-episode-reward.png) | `env=SpaceInvaders-v5_type=ablation_model=c51_metric=eval-episode-reward` [[png](./figures/env=SpaceInvaders-v5_type=ablation_model=c51_metric=eval-episode-reward.png), [pdf](./figures/env=SpaceInvaders-v5_type=ablation_model=c51_metric=eval-episode-reward.pdf)] | Ablation study for C51 on SpaceInvaders showing metric `eval-episode-reward`
[![figure](./figures/env=SpaceInvaders-v5_type=ablation_model=c51_metric=total-model-parameter-count.png)](./figures/env=SpaceInvaders-v5_type=ablation_model=c51_metric=total-model-parameter-count.png) | `env=SpaceInvaders-v5_type=ablation_model=c51_metric=total-model-parameter-count` [[png](./figures/env=SpaceInvaders-v5_type=ablation_model=c51_metric=total-model-parameter-count.png), [pdf](./figures/env=SpaceInvaders-v5_type=ablation_model=c51_metric=total-model-parameter-count.pdf)] | Ablation study for C51 on SpaceInvaders showing metric `total-model-parameter-count`
[![figure](./figures/env=SpaceInvaders-v5_type=ablation_model=qnrl_metric=eval-episode-reward.png)](./figures/env=SpaceInvaders-v5_type=ablation_model=qnrl_metric=eval-episode-reward.png) | `env=SpaceInvaders-v5_type=ablation_model=qnrl_metric=eval-episode-reward` [[png](./figures/env=SpaceInvaders-v5_type=ablation_model=qnrl_metric=eval-episode-reward.png), [pdf](./figures/env=SpaceInvaders-v5_type=ablation_model=qnrl_metric=eval-episode-reward.pdf)] | Ablation study for QnRL on SpaceInvaders showing metric `eval-episode-reward`
[![figure](./figures/env=SpaceInvaders-v5_type=ablation_model=qnrl_metric=total-model-parameter-count.png)](./figures/env=SpaceInvaders-v5_type=ablation_model=qnrl_metric=total-model-parameter-count.png) | `env=SpaceInvaders-v5_type=ablation_model=qnrl_metric=total-model-parameter-count` [[png](./figures/env=SpaceInvaders-v5_type=ablation_model=qnrl_metric=total-model-parameter-count.png), [pdf](./figures/env=SpaceInvaders-v5_type=ablation_model=qnrl_metric=total-model-parameter-count.pdf)] | Ablation study for QnRL on SpaceInvaders showing metric `total-model-parameter-count`
[![figure](./figures/env=SpaceInvaders-v5_type=compare-best_metric=eval-episode-reward.png)](./figures/env=SpaceInvaders-v5_type=compare-best_metric=eval-episode-reward.png) | `env=SpaceInvaders-v5_type=compare-best_metric=eval-episode-reward` [[png](./figures/env=SpaceInvaders-v5_type=compare-best_metric=eval-episode-reward.png), [pdf](./figures/env=SpaceInvaders-v5_type=compare-best_metric=eval-episode-reward.pdf)] | Comparison of best models on SpaceInvaders showing metric `eval-episode-reward`
[![figure](./figures/env=SpaceInvaders-v5_type=compare-best_metric=total-model-parameter-count.png)](./figures/env=SpaceInvaders-v5_type=compare-best_metric=total-model-parameter-count.png) | `env=SpaceInvaders-v5_type=compare-best_metric=total-model-parameter-count` [[png](./figures/env=SpaceInvaders-v5_type=compare-best_metric=total-model-parameter-count.png), [pdf](./figures/env=SpaceInvaders-v5_type=compare-best_metric=total-model-parameter-count.pdf)] | Comparison of best models on SpaceInvaders showing metric `total-model-parameter-count`
[![figure](./figures/env=SpaceInvaders-v5_type=compare-best_metric=train-episode-reward.png)](./figures/env=SpaceInvaders-v5_type=compare-best_metric=train-episode-reward.png) | `env=SpaceInvaders-v5_type=compare-best_metric=train-episode-reward` [[png](./figures/env=SpaceInvaders-v5_type=compare-best_metric=train-episode-reward.png), [pdf](./figures/env=SpaceInvaders-v5_type=compare-best_metric=train-episode-reward.pdf)] | Comparison of best models on SpaceInvaders showing metric `train-episode-reward`


## Authors

- [zanderman](https://github.com/zanderman) [![zanderman github](https://img.shields.io/badge/GitHub-zanderman-181717.svg?style=flat&logo=github)](https://github.com/zanderman)
- [saadwalid](https://github.com/saadwalid) [![saadwalid github](https://img.shields.io/badge/GitHub-saadwalid-181717.svg?style=flat&logo=github)](https://github.com/saadwalid)
