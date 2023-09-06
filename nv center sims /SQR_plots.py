
from SQR_sims import *

## the list can be plotted vs time to get an representation of the
## signal, on which we can perform an fft 


plt.plot(sqr_time, sqr_list - np.mean(sqr_list), label='sq ramsey')
plt.xlabel('time')
plt.ylabel('sq ramsey signal')
plt.grid()
plt.legend()
plt.xlim(0,0.5)
plt.show()

#fft plot as well

ffts = fourier_transform(sqr_time, sqr_list - np.mean(sqr_list))
sq_ramseys_x = ffts[0,:]
sq_ramseys_y = ffts[1,:]
sq_ramseys_y_mean = sq_ramseys_y - np.mean(sq_ramseys_y)
plt.plot(sq_ramseys_x, np.abs(sq_ramseys_y))
plt.grid()
plt.xlabel(r'frequency, $MHz$')
plt.ylabel('')

peaks = np.array(detect_peaks(np.abs(sq_ramseys_y), show=True )) #mph=1

print(peaks)

ramsey_peaks = np.array([sq_ramseys_y[2], sq_ramseys_y[6], sq_ramseys_y[47], sq_ramseys_y[88], sq_ramseys_y[127], 
                         sq_ramseys_y[129], sq_ramseys_y[168], sq_ramseys_y[209], sq_ramseys_y[250], sq_ramseys_y[254]])
print(ramsey_peaks)

