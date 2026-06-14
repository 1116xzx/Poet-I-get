# runs mulu shuoming

```text
runs/
  moxing/          san ge GRU moxing de xunlian, pingce, quxian
    jichu/         baseline
    jiaquan/       weighted
    jiegou/        structured

  duibi/           baogao zhong yao yong de zui zhong duibi jieguo
    biaoge/        csv / json / md zhibiao biao
    tupian/        ppl, geshi lv, cangtou lv deng tupian

```

Zui zhong baogao yong:

- `runs/moxing/*/metrics.csv`
- `runs/moxing/*/evaluation.csv`
- `runs/moxing/*/*_curve.png`
- `runs/duibi/biaoge/san_moxing_duibi.*`
- `runs/duibi/biaoge/xuxie_moshi_jiu_zuhe.csv`
- `runs/duibi/biaoge/cangtou_moshi_jiu_zuhe.csv`
- `runs/duibi/tupian/*.png`

Lishi tiaocan wenjian yi qingli, dangqian `runs/` zhi baoliu zui zhong xunlian yu duibi jieguo.
