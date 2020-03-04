import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

n_groups = 5

means_men = (20, 35, 30, 35, 27)
std_men = (2, 3, 4, 1, 2)

means_women = (25, 32, 34, 20, 25)
std_women = (3, 5, 2, 3, 3)

step1 = (17.192,21.697,27.746,30.482)
step2 = (17.754,21.904,28.16,32.241)
step3 = (17.369,23.452,26.935,32.967)
step4 = (17.613,23.496,30.492,35.642)

step0 = (12.441,11.451,10.441,11.922,12.354)
step1 = (17.192,17.754,17.369,17.613,17.829)
step2 = (21.697,21.904,23.452,23.496,23.972)
step3 = (27.746,28.165,26.935,30.492,31.326)
step4 = (30.482,32.241,32.967,35.642,37.426)

fig, ax = plt.subplots()
# plt.grid(ls='--')
index = np.arange(n_groups)
bar_width = 0.15

opacity = 1
error_config = {'ecolor': '0.3'}

rects0 = ax.bar(index, step0, bar_width,
                alpha=opacity,
                 error_kw=error_config,
                label='1h')

rects1 = ax.bar(index + bar_width, step1, bar_width,
                alpha=opacity,
                 error_kw=error_config,
                label='3h')

rects2 = ax.bar(index + 2*bar_width, step2, bar_width,
                alpha=opacity,
                 error_kw=error_config,
                label='6h')
rects3 = ax.bar(index + 3*bar_width, step3, bar_width,
                alpha=opacity,
                error_kw=error_config,
                label='12h')
rects4 = ax.bar(index + 4*bar_width, step4, bar_width,
                alpha=opacity,
                 error_kw=error_config,
                label='24h')

plt.ylim(0,40)

ax.set_xlabel('Input')
ax.set_ylabel('Output')
ax.set_title('RMSE with different steps')
ax.set_xticks(index + bar_width / 5)
ax.set_xticklabels(('1h','3h', '6h', '12h', '24h'))
ax.legend()

fig.tight_layout()
plt.savefig('static.png',dpi=300)
plt.show()