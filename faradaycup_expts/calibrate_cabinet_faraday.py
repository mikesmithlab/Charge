import numpy as np
import matplotlib.pyplot as plt

faraday = np.array([3.25, 5.01, 3.75, 4.46, 5.06, 2.22, 5.00, 3.06])
cabinet = np.array([2.99, 6.70, 3.96, 4.90, 5.66, 1.99, 5.29, 2.66])

mean_values = 0.5*(faraday+cabinet)
diff_values = (faraday-mean_values)/mean_values
print(diff_values)
print(np.mean(np.abs(diff_values)))

plt.figure()
plt.plot(faraday, cabinet, 'ro')
plt.xlabel('Faraday cup current (nA)')
plt.ylabel('Cabinet current (nA)')
plt.xlim([0, 7])
plt.ylim([0, 7])
plt.plot([0,7], [0,7], 'k--')
plt.show()