# QnRL: Quantum-Native Reinforcement Learning

<!-- Badges -->
[![python](https://img.shields.io/badge/Python->=3.12,<3.13-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![JAX](https://img.shields.io/badge/JAX-0.6.0-7E0B8C.svg?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq5MTYAhOxXM%2FmvEpsoWmtfP9RxnkAIAr%2F5pVxqPxH93msKodRSXIct2l0OU0%2FL4eY506L%2B3GyJ6UMEZfjjCDbysNcWWmFweJP0Jz%2FA0g2gk80pGkYAAAAAElFTkSuQmCC)](https://docs.jax.dev)
[![Flax](https://img.shields.io/badge/Flax_NNX->=0.11.2-1FA89A.svg?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAUkklEQVR4Ae07CVSUR9LFzDAMp4iAnCP3fSggIo6KBhWNB4lnTOIqGo1J%2FqyJGn9%2B13V5ibvxGTYmG3VzrRqNt67xVhQBBVGQSxBmhvtm5JJLmBlm%2FuqBgZnh%2B4Yjkry3b%2FvNvO%2Fr7qrqquru6urq%2FgD%2Bm34fDcjlck5bh3hb0%2FPOPfhu%2Bvtw8Tu0isIyJNLuyIS0Kv66mHj5yuhb8suJJVWdXdIorGP91izp%2FJYNooATnxY17T1%2BXTA3Na8O5CqN%2BzqZwVvz3R5N9rbcwWDoJKhUjerrqCsAhSYCmNc3de4%2BdUu48UpyObtL0k0pFIupA7MDbKVvznc9x7U2jkagUh2d0WVxVKmj8Jz2F5INN1MqYo7fFJo1tXZRCq5ZaKDHghWvOHcsmenwhamJXiwqoUUT5mXlR00BMpk8IiW7du%2Bx6wI%2FfnnziPi1szCENyNcS%2BdMsd%2FFZOqcQEXIRkRIC9KoKAB7PubHf%2Bf%2F6URcIUPWMwW0sKC9ijAYHmQL%2F7su4DiTyXhbO%2FTwa0fL6u5dFu4MDKbOx%2Bfii406uqTD5wwxdFkMWBg6QbxqrssPaBh3j4jIIEijMgKUbeJIcCssfx579Cp%2FYfKTWhjOYPB3HgfrFrnf93c3%2FyPSyxgtYzhiBRDrjr8w0AFzho7OOaXQms8eOPnixMc1%2Bw5fLnArF7VpgqjlLUw5sGa%2BW3VEKDeaxWIc1zbvkTYPkc3xf0kbnFoDGhmGRn5IWWzYoay69fQXP2XG1Yjap2tDIj2H%2FyszA633vrPeRbo6whn02QNnni6TAa%2BHOcC7G9wggmd%2FCIU%2FNQShPPafyDnPL2m%2Bijz5aOODrm4gJ3SQWI6NmDS3dG09epn%2F8dn4IiOxRAar5rrSYiA8cXYCsp7VfnkwN2PG%2FboqsDc0hj%2Bs9QX%2Bww64n0OmhRwmuo6D0DBzOCnKhYNpjeBfZPHpB94By7tlso9wdCVoU8SD3DrG9QflEZEzHGfXNXT809JM%2F1OEr6dlSqNiSFMAmWRIpbJV8WlVe3BZc6h81q4gQ3rtx51hX3NtjMk8VUuIY1Pd1rrraEFO1OliPrtL1u%2F8kEZnWdtDmK4TdIvlkM2uhssVRdCNylAmJo6cRVxn2XpP%2F3MupmY7UahCZZ3yiW1sWPl%2Fcd%2BLml4oiszHcOCtCFfRvKnc3Rw95r8QR6yEpXtqHQGkd%2FAX9ETYQAzZjAxBPclrTYjDwZ7bdlqQt%2BPg00wjUWcPc6pIhER8TQU8YFYDS4cBrVKJarXinSjjYlkhI766fMU7Hn6LZTLZ1ygQ6d0%2BI%2FKiU2og7e53Deqfd8L%2B008sb6ZWHFq70OMdxNmO8PH4H0BfWUCrANKDVXXtn569XbTm6oNylmpDSmTVJ8KTNX%2Fh7fLifRn1tW5kqHOYtOQVqC%2B6yajoHxmq9Mg7Ydt9zFjgGhpzttyP%2ByTczmG1WCqN1mUyiX2QSqQydnf3wB7JL2uG6IMPA14JtI1bNc%2FlAvIWjfADRhBpg5JD3JlFnrlV9OPJOKFZc9ugo4jMY5%2F8hmf7DuVlzr1TU65wfj4L5FWfCF%2F805nC%2FA8P858YUPUyYYAujefow%2FveAaJFjq67rpQId96qKuPGVZXZXSwRHnvPe9JmbHNrixbeiAMWl17JSMmtXbZsllNES1vXVhMjve8026NcBRqaO6f%2FeDl%2FUOGNDXSBoSePjM1Mfbw6%2FnJEXHVZn%2Bcnk8s6x%2BkbRG%2F2DfQ%2BPWfJmWUOrlIWg7I5NZ7YDCasdfUWn5kbeXC5q5envi77OxRW4UmRvk4RVcO6hOuhf3mUlNza3bl5rLGeGr5mpr1TCkevC4yKK1vnaNaRPOUIUAAOHFl9%2BCw0fnOCbcHClwlr7l%2FmVnf0GMU%2BgN6X3rlXigKs2x0wnTFPz3bZ4SYBPEAhNMmT4c4bbwt%2FMHSGyZ5OF9kGujsQv2%2B%2Bq9KWyGWAhpVxq7LU6e1F3hBUbA5X7pVDp5h%2BOqniq77TK0AVSuXdz9kMeDMt4GJzAWTwRSo16q9omlgoOEMuky8uflC2L%2FVf6S4NRY241jvCwvlT4Ie6Aihue65AsjcwgndtvIF9pQ4eJyZCkX3WiinrgoK7pd3RDCbjzFlBnjrx3lyTuAu%2BFmSAq7EprFnrDQWP2uFeds2ghlqV2LAU4DnBFBzD2RDDv6e2ZKkSVL7jMmYuEtbfTDueGV6YWIxORE9N4Z0iYCWXwfalXiCc5AgdaAQn5jEg71A6SDp6VoMmNGI3%2FnLboWAq92TwmoD3DfXZWkNmwtZm%2BJSfDPM8J8A8jj3cSK1QsjHoc1gK0NNlQlbjM63CG7N04X2fAJiQIzU49%2F2lcMmLgUucFOdl9s85YHjTEEzYSLOaertf%2BqAcKjOqeQFrJsKeqdPh7zlp0NDVSSkU0W9GfR0s5VhT1tMVDm6V6DA1ylm41i6xd4ZYziRgfF4AkuYu8F7sCbr6uhqQ%2Fdn2%2BnZooRGeQDH1mOC5wB10cGq3xWTD38ResMbRE%2FSZzH4iv%2FJtWCOAri0bA0P4xMwXWs%2BWQkamUAHWWNIEpvZjYNp7IdiLVVCUWAJoD%2BhIqJejRZwwxR5cZztD1uknUF%2FUoKhvOpgOXJdxEPtGEByQCiGvuVEdbwS5lzIC3EzHQeNxIVRlVvexgMYLrHysoA7d5soWMQRtmgLmrmTjpj2Zck1hyrshIJLIoeCJCOxD7IFtxO5Dqi9sgOJvc8B3zOC0%2BpC0vAx5BIwxZMPsIFvp%2Be5mahyVzrX2swKbUAdIucqHUtzwkJSbUAIhkZ4QNN0Bnv7yFDoaOtTY0sP13AunjOh5F5z8ezJI0E6QNNbaGGa%2F4QfiymYQxPU4c9rc8SAPC1lSRg2joYXaVqg1ihlqYVSgWAwdiAjhylbMcT5ub2V093x83mGVasrXgLcCoDj%2FGTyr6FnmCFC3VAbJ5%2FIge6w%2BzEKB9Lok8PRyAcjQl3ef4wpMFPTmqSfQVNOqRlOKa7sM7YvrXLc%2BBagBaGRC%2Fa3%2B6mxrYnrhbsm7F5NKWGJsV1uiVQDZPwS5W8DbC9yS%2FNzGbUcij%2FAfRkeMgeErZRLinE86nwfz1gdCQ1UrpFzIBUlXj5PShju3ywcfgjX6E2FRQYChLki5JoASFF416WJkOHSpN4y1MoLbhzNwFPj2VRMHizhONKluvLnBrs3LvQ%2BFB9vGHrsunJucU8tAn4QSnFIBhmi5d7w9qZgXYB2NcbkL2GCPK0pDhFD2X%2BoDJlbGIIwvUjT0XNQOF764D1xvS3h923QowLU5B%2BuUJGrQKTq5N4kES9SNI0rmg9PEe4YDpF7Mh8QTdWqMW3pYgO8SL7ipRx9pJjQxPUWhX43ZFLTgfmbtXpzCJmqEejOUChhjzD40K9h2NxKidEVVCZno6sIrNlxZ5hc5DBb2Gu%2F9qdBOlE14wGd5nggqcDr4hjnCyj%2FNgmQcGRVPez1IrFftmfGOY2HmG%2F5Q%2BLgKzvw1UU0xZDkN3RQMzZUtkHYsEwI%2BD5GZ6QkZjTR%2BAeER%2BSfj%2Fwq2cRufHqRMM1EqABF7rI0mNIBRZ%2B%2FeXRc3Nku4ztIoT%2F8LjmPGJp%2BU539VnVUDNRj8dHvFBZZ%2BNA0Sz%2BRCfeVzhSA58cWQn1wO05Z5Q%2BA8V0g4kQ3NdT361TfRg1lv%2BqOdkMOlr1Kgs71%2FB2qAdbxlPsB6IYb0S%2Fkgxh2gIZ4X8Kzsvzk7Z4nBUX7umjPFBexOxdZ6IMOkBOUhFjGLqpZSAVSAvWWcCHtHMGXrQZSH36NgK9utWH4f%2FwuVOHLcn%2FNvCYGNq8YrizzhOQp19%2Bcs6EI3V4Lh8YSfs8F4nAHMXO0P4g4xtDV1gg2GxBJRIapGkyyjU1%2FzBAcHU%2BBfLYCyWvXBiD5FjY2RyefRgaEHFk1w3vd9fnb4k3r6vYmSP83ncBXAKG1tgded3DsmWlgdQGKpqF21YaxsQIy9mHUqG4wsjWDF%2F4SAAKdBOho7shq04hJ45R8PwDXYDhxxyTyHtkCZdNAoeqMNmMTjQnGcENLv0A1GRc8StCwc4vO%2F5M05kFpVvlFJZ6jPftM9NIz60rYW2JJyx2Bz4vWjaXXVidg4TxtqG4bBH333CDiNHbBqGw9c8JRHmZ6jk0SUoUzEYK76ZDrYGLAg7dBDqBf2eIDKes0nsR%2F4D80Q1dz5IOnmxp1p9zRBBs1TjgAkGoKYldi7lRoUFIsqsXEpohpIq78eupjrnPieT2CunnG%2Ft6aBo8jW4nF4XYEIpqyfDIXpVQNAmLiMhi3xgLQf0kCG0ebBEo4Ux6LmxrNH%2BU8iMXbIkshkYMnm0KKhTLhd6VnNVIEoFdDY3Ln8bnpVVHuHJMZAn%2FXPXiOiiqd4J42eLxUyMFTlt2alB%2Fh7mkP%2BhR5DNQAYC4h9kGo5JutAH2Ew4fXRkXJf7gnfFmRuPCrMgzaKgKpq2yi4EYbyP8bwmSeWv6FaR94ppwCJrHx3Md%2F04%2F0pXz7KFSUjER7%2BNXH78i0SMXxTlAOfW5aC%2Be5J4D7fDchcfpmJ7Ax9l%2FuA3p98YBenAA7kZ2kVHvll4An1qtupldkfxibH5Aobzaj4oRwBSkByrE2iqxEh9omr57kcsR1vdEdZR%2FUsa2%2BFPxenQWCgJWwMnAqZn6VQgQ2vDPXogrtCyXxL2C%2Fig7CQ3gFSEkbhg7P59egFCniP%2BdrPSLQqgBAk0dVrD8oZeAITtXaB%2B1uBZpbw%2BFkNSLWMiMcN6PzY9hs7JWMjeZKdYNVCU9gveKgMKtGSGavHAbyJ8kc8w3DAswGWROXMgA5pUAUoEcntji9P57C90Vv78%2BzpcL4hH7Kbnimrh%2Ffs9RKHgkS8%2FucY%2B6OfgAAGeP6wCgMlls%2FGwPrP7rq09IbWVOmLpd0Gqnnl%2B5AVoETIw0CH4MhzmB%2FqCLM9JsDR0ifQiAwOOWFQhM53GDKNXkByfDbHZgKEABeuX6mCa3X0jhDaA0pZKY3gYIyQoXXpXhncPVcPqx19BgMfUj1RynDN5ioHDzDIMYFvT%2FKhrNetHlJjKkAjUoASv6GlC%2Ffz1IOTbF7IkqWZjEz1wWniwMCljds4YLEGxvpYHBbtisLBcEYWOktazJFm8wPyv0oBA6ipFLBM2bUrDkYe8XvNW6YaK5Ci%2F%2B%2Fvawkro2fCOFsTMDLTh8gtocCb7Yh%2BQr8DRJZRt3AXeO2LV5N0dBlSFdIv9ZVyXryMFvCEuMPUxmTdrC28b1GQ2EdHH4dWpFUpIkBPrxQoNkshuCHimHDg4eF06GoVg89rXoqmx3taQEjUZMGEyXZbcWrkdIgkPUGGl8GYBo1frQBiiOgSmdeYUnFdnrVk7%2FxjWXcFK3Q7e6ZMF64qKd8%2B6kMlsQRD3CXytk6DiQu8LqFr%2FAbidyCuXR%2BQxou2tjVAabP0U4BeLjVii7muhVt8AttMdNX3Ani%2Bz0bmiR8RlCWqiXsv6eay95ofQqJtG0xcHwAkCKpMTDwc8XrdC4r8WbBNng07UuMXop%2F%2FM%2BI7KWFUn%2Fq47G3y8BOvdPGi3yqqImh5pxwBFmP1b25Z6bfs2A0Bt0Zlt0ZFRy7VubbJJyA2guv0tx%2Fys1f80rsxwX42LWtpPvoT%2F8lbZ4sFDHKgSdKBgiywNjaETTGTYMztRhCjIdVZbAv7GwpAkNnj5V2tLGEk1FZGbnD3nbva3ecnZbuKZc92gmyT16RrHmbm2ytq2jbjHYEPlfV0z%2FFojK3NDdUDCr3AtP2M2jfFY%2FIdvySWbjkXX8R5QXHyqnpFhvQ2Cs1Lq62KxXsCwTOs7OB7fg6QA0yqRBoONB8PRniUloR3h4jHSZUCxlmCxxgzyMYjObwX8DTMzmE73hu6gdNDVl7d%2BtX6PQkf0nl8%2BjitloY5di6Z6fi1%2BVjOXsQZcJJCOQIIIwjcjEJFRy3x%2BHFWkM3eY9cEkUlZNYxumtMdhCdo9xFnWuB4m6i69taY0yUFVnQKIOKm41neYKldIoGVLp5tn5hM3aPHYu3HdgYN%2BDNxBZkdaCvFC5YXnexMorGNwl7%2BBjRHqwAC2YtUiEIt3bUhKDw1p3YfKmJivpa7v4hDlixyqeHMifAl208J8rYcEeQO%2B4aIHl6UWOHkLl7r4XsEQ18xSLd6APcUBf54dPbmPNf0yT6WJJSf0CsDBWRPkVYFKLF6idxGoaZMdDffcPNBxe7jN4SWLSrBSyWs8ok4ZELvRJzDc%2B0d96J9iLxcUdx3g0QJp%2Fkk44hMn3e9JyX4W1htxfyQbona4Ary9ny3ytnBtjvZukxysXpIvsOQFKBkEomScO1B0ru8ida7LsQXb9TVpV9ICB7iKEbQ56GzX19a43H6y9w0VibuFqmSq4kpfOgVBOEOjuTa3TeI2%2B8ZUSFgmTF6nHitviUyzHG%2FiRF72FfrFROXhrbWYlQCqffDvx0yeo0OGOEYeJNr9a0HFXtO3BJy%2FVzNwBKv1hzBTVRtZ088kCyha3FPIS9mw73MWrxF4lS%2FaIbDHrzrR5RA25MdLySxuP21tLc22oVwpXQ8aCsfsQK0ESV1vQoKyBE0fHXkCp%2BXKewPTJBeWzrHAerMmzGqIwbPF1ZwMa4MRM399s3VbgxELXLPCfYZ%2FxH5hIZqNBDlYlN4dDhyMUaOqUUDyJg5XluNOR1XtOFKchkb12pKaAc8SiNLVX5ZE2U9seazAmxl5BMaBxtjcumxnBLwVxRqn8AjJ7ws6XFN1LWUclrhCenS2lZa4Uk9WXITMqsYN5LLl2F2Eyl72WlURgBhEkeBF97i3odfiEUk44Xm3ikxLP7JNwNrXnVLCfC0IJui1GEhDxF41BRA2idzFK%2FYLkzOrI09cpXvQnp8KMkSYwZrFrjVhk%2Bxi9ZjM8k3A7SGcCj0tMGMqgKUDaMiOB0vpFv%2Bfbc4%2BsydIhOqmB2B5eCmCK%2B9d66c4%2FwNfi22BwUfPASsbGSEz99EAUreUBHc8prWPaduFa6%2B9aiyz60mTEzDM0L8Quyah%2BNYYuyeKnH%2B457EFnR3y3hpuaK0D%2FYmycnns4npVXz8HmEhmTL%2FcQLTCYTCsrrE3Rvw%2BG0bvhvQwf23fJQ18P9b5ZkNm5648AAAAABJRU5ErkJggg%3D%3D)](https://flax.readthedocs.io)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.42.0-FFBB34.svg?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAAA%2FCAYAAABQHc7KAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAMI0lEQVRoBe1beXCURRb%2F9TdnwuSAXMgliEFuQWLCEVBJAGFxZctdXEtc3UKr3AUKj6WylrqILOVdWC4r6lq7rqLG%2B0BYEcQQAbkPkYSECDEcQiR3Mpm79%2FWEzMw3M98xudaqpf%2FIdL%2BrX7%2Bv33vd7%2FsCXGqXLHDJAv%2FPFmChi196hBs%2F%2FhFzLrgwy2rAcJcP1TYjdmTG463iqaxW0PJDvzKjpWQePHU3QLJeCa%2B9CsbEHbBd%2FS4b91FzqLzwfk4xH%2FyDHb9v8WKsjyOZhxN0w1gs0MBgN0soTzHj7bI8tid0moABsrbxYWXNeLPJg6xQAtG3SrgwIA5%2FOJ6QXQHHD2%2FSokeG00CynIYpZSmbUv5hOO72%2FRzbavCn8w6sdHNYw%2FE9NZZotSkmvDs6AQu%2FymX%2Bh%2BU3wDVFvG95C%2FY3e9BPSZkxNS2%2BAy3z642jivso0QDMB4PtfVgHLWU5u8%2B10%2FXbxB%2F40YHneuKJt8%2Bp9ptswhdz0zF7XRbzSYKw0o4n1BYvrPTkB2WS%2Bx939uH2JBXZXIK3aT5ajx%2Fl24cu4KWLMKGID6p2YuXPZfFC%2BXo3Zm6vxV2iL43Zym3NXvxWDJTalXWtmHbkAnhtEtzrf6NEFoT7XH3gqn4D1R9uSGj5fpmHIz6I%2FHn0fnJhodBEMhswkYKdql%2BOO9sEiaKWaJ7N%2BfD9ONjfV%2F9D9N6mOe%2Bt27B4yY4qmC7yq%2FP0HJbWPFrMJuKCTWtak9cXJHEZ4X7nTkoH%2FvARhCv0EhoZnnyrFJvW7MWo6hYFqp4H04rMYlbxYA9RIAhZYaQyx9J6yYDeg2Pg%2FS5HBlMasPRTflROWR22PfUN%2Frz1JCw%2Fg91Ame20UEw6cD2rjDPgkNICBPxIhg0V%2FUM2Cu1u19t3kD%2F4jajGCkPObkoObSRxDi8e%2FaAcX63ejQnkVv%2FLlmDEZ2J%2BfxZINWOl2oZ2k6M89stM%2BFiQip%2FOgHvrTZprMGQehOHaAzK6q0804Mund%2BGvGysQ7%2FbKcD0xoENR%2FQgbnhFzGcSfKYseO3beiWlOH4aIcbRWlk5uQPsmt6wWrC0egp%2B8EsapX4NZ7NFY2mBkM8PYg%2BBVV4GfTwvQGcgNJh2vw81Hf8LRgQmoSlaNwwG%2BruhcZsHyb6ezz4WswCOldDi2pBn7vRxGpUkE8byj1Xj2nVL0rXH4yYx5xTDf%2BYISSxDuNcHzzQx4vpwJXyWdt9qOIH68x8jw6vTL8fjsoWgyR5meDK4apIKzaPZ6GVCR0xsjtk5hHkEcMIAY9NnI%2F17rxh9FX62l2t1Ytb4ct28%2FAyZ5YV3%2BF0iXH1NjCeIoe3BnIuBIiMgkjTZLfVnCxCWPYO3WIAPFG1p9nRujT9qxjI7q%2BaG4WPpisUPicdOJGczv%2F4JXZgC6DyQfacRJcoVkPYJvLK%2FB6sISDE45BGvBIyStC54To7Ak2dbA0q%2BATdwn861b93Lr%2BvP42u6NvK%2Fo0bcPHYFr57BZobT%2BINgO2Hcdq6cb06PtY63fz4elYHLBJLzcfwacB6drkevDczKit3ExXbp28z2Th4UyvXMtc8Qb8EYoTG%2BfvMzVz4ql4fQyAwjk1Yl4iVJESTih0rjOYsR984bjxsueg93Q96wSXcxwX%2Bto2I9%2Fwg%2FMDsm%2FMUsJMPQ24cXvprMIP40wwH8mMU%2BGBUsjEAFR0TtFtisw3nm4EOa0JykwuKJTxQj1tQ5HS%2Bnd7Vzz93KJ3JMOILE12jW1oxKwMhpX1HVW5LMtiSZ8Go1BDXbSGX%2FvGNeJtbD0nwBDvDz5qzGq4bwtgdvX7josiFavUGMXOHr6BUW5bQWdcNqoBhBEmb3woElCTE%2FS7UP8KQeeYZNLv4Mp41qYUpeBmWKSEa4gRVb%2FpWXuLp5c7cITkXh1CLnzoUm98U8lKkUD7L2OVaSasEaJUQlOdYX5gzfzXDb5iI9NrXyWovko2g3FSvTacF88%2F%2F5xaWctHmz1KhdsoskRNz0KfPe%2Fn62cnhQNIASOTcQKy8VLQ7QJosHoIIUaF%2F72i13cf6Jhk49WIH1%2BHszpS2g3yNJaNP4IGHcbV9dMntbowQMROA2AzYC3qAZYpEamaoBNk1kjBUTdabF9IvLTcQcacFf7mI1Y42G5J9YgbugoMoLqxaudJ%2FDLOd6%2BcNVTsRZV6LzvoEPPQwE5Ch1VAwgeCiCvUWV4lwK%2FIrjWhVXji7jsQEUHm0rEZ94GZvAfQxWZQxDe0glgZ5KyQ0C6uuT7qw7fwKq0iDUNQEJAu2AJ%2BZNupcWklK7S6ej6cLgCLGfPMUhx%2BgxK1203XbutFFhiaXRnq8xOxvN6eDQNIIR8n8%2F2JRqxTo%2FAUBrSe%2FGIL%2FnwUJi%2Fz4ynI2BRAJ7i2f6LU5IzNgNcZsVDdJ7xl72jiJWBdBlAcAy0ooCOk40ybo0B%2Ba31NJXDI8i4uzoCFgbgzX3g%2FugWPzTFod8ASSZsn5mGwjBxikPdBjgynVVTPT3qaUpROiFoF8wZ%2BAWfI6NhBs2n4954C3hDWzHZatdnAOGmGWYseXmc7I4nmzp8oNsAgnFCEl6g8llFuBC1said%2FOTEM9nbeLB%2BxnmNGg%2FcVni3XRcgsTn1VY2SjHitPJ%2FFlGViMgClRVeaGQ%2Fqt2%2FbGhw%2BjKSAuDiwIp9d1ZV81QPAm6wB8hQdMUCUuUYmxJ6yYzKA0KhqJvuU0uKWgHY6O3SQWT5kM0%2F1k5upIqTSeEucDGtt1XYBStcrdkxl52SMOgYxG0DIpJraUnrjGtMZn9JiYoO7rRAJ7lHNAiy5QVaqsWkEQSpzHbshFS%2FpWG8ESYcMQH5WQv72SoQ0DUCDBwsyt%2FAsSOoXJCn1LFjfCwFp6SoGEO5IaW9ZYRZrK1IGuPR1OmQAIXpsEpbTPaFW3zRtVKLges6J1Q7LMPXdI3lguumTgOheVINUapSZPqbg%2FJkSXgveYQMUTWG15Hcx3xMoLeYubFp%2BvZZiximbYMwv8pPFKcQA2vrfUn3%2FHiqVaYlTxHfYAEJifipeEUooSo%2BCEGlxZ%2BOQh7lT%2FrotgpRusOY7XoR54b%2BQllgjq95SxG8m479A5e2pO6exoK9ECNEGdNx0F2VfsZlPpxS3mRam25jiaHt%2B132wzP23toZEwZFwdqLh7M308iadFu6g097Obbkd8%2FnwCTttACEweSP%2FiD46mBcuXGls8%2FhQtWITkh%2B5Hyz1jBJZEM6MFzBgUQbLXNUFdfegWNHT%2FdTkbPIR%2BeGyWMpnTsqhLR4rXO%2F9jgTpeAbMlArnGX%2BBRT5z50ddYoBvprEK%2BvhI1%2FVTqOyml6yt9Bmad%2Fc18B4bp70KTgchydx2MdCmjomiSwwgZhxuwwoqP%2Bs%2BibX2MtELPwZ3Ie0Cem%2Bo2jilQceZdFWaDiK7zABUdranW1CgV48aeqEimu%2FEQHi2y95WRRehfYOOzqcB7TIDiHlu6491FKH3aczpR9dZgy7tfv%2FX4PZkdTYpXv0Coc6tiO1SAzwxkvn6WrCI7uWa0brBFtz2vMEG9yfzFZX0I5gUe0VZXaIf26UGEBLFp6i9dZTPSvraZOp5tuTBd3aoDBYYiJcrloEx1SECvBqdLjeAmG9wPArotKZ65%2F9gfAac9I1eoLkNFBDvoFNPFJUMvTayMa%2FHdO8IyNXoRJlNg0MHev%2F17BwVUVXLZyeT4%2FD87CEyad7Do%2BH9dqIMBmaup2%2BQ75cDu27ULQYQ6s1OwxoqT5erqbpqxhVYO2swPfSLhyE6T7sKF9DXZ5Y2NslyDqbkWWzS4Uo1OZ3B6TiGdVz8oC%2F43FOtWC8uQEpNKDDph3rcW3wK2ZX1EAVQ692FlRlZG9bCOuQllr1d1ZWU5OqFd6sBhBK2DXyDqAzrUUgoI9GrsDgj%2B6x5LtP%2BBk%2BPUA2abnOB9nnpfcIyveUzsVO8dEym30Ht%2FN392%2B0GKM1jJRQQX4xlIVQ%2FrI%2BFvjO03W4AodyIBKygtFitV1Eqc5Xope0sXY8YYOdUVk%2BfqOkqn5G7%2BOjtztrOLkwvf48YQCgzNwOv0gcLn6opJoIgldyfPprHYiqzqcnUwvWYAV4dz3zDbLiVSlqvR0s9IlBSefvh2wdEvlLXWkRn8NF06Yw8Td57DnLsqUdWVSvE11%2Bj6b%2FIGukpHB5gxesleazrvjPU1OQSwSULXLIAWeC%2Fb86ZlN1ZcAAAAAAASUVORK5CYII%3D)](https://pennylane.ai)


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