
clear; clc; close all;
set(0,'DefaultFigureWindowStyle','normal');

%% Simulation Parameters
fs          = 1e6;                 % Sampling frequency (Hz)
N           = 1024;                % Samples per frame
numFrames   = 1000;                % Number of time slots / frames
SNR_dB_list = -20:2:10;            % SNR range (dB)
includeRayleigh = false;           % Set true if you want fading
pauseTime   = 0.1;                 % seconds between frames

numTimeSamplesToShow = min(200, N);  % <-- avoid indexing issues

%% Dataset storage
% Columns: [Energy_normalized, SNR_dB, Label]
dataset = zeros(numFrames, 3);
row = 1;

%% Prepare figure for animation
figure('Name','Single-Channel Spectrum Sensing','NumberTitle','off');
set(gcf, 'Position', [200 100 900 600]);

% Time axis (for plotting)
t = (0:N-1)/fs;

% -------- Subplot 1: Time-domain signal --------
subplot(2,1,1);
hTime = plot(t(1:numTimeSamplesToShow), zeros(1,numTimeSamplesToShow), 'LineWidth', 1.2);
grid on;
xlabel('Time (s)');
ylabel('Amplitude');
title('Time Domain Signal');
ylim([-3 3]);

% -------- Subplot 2: Spectrum (FFT) --------
subplot(2,1,2);
NFFT = 4096;
f = linspace(-fs/2, fs/2, NFFT)/1e3;  % frequency in kHz
hFFT = plot(f, zeros(1, NFFT), 'LineWidth', 1.2);
grid on;
xlabel('Frequency (kHz)');
ylabel('Magnitude (dB)');
title('Frequency Spectrum');
ylim([-80 20]);

% Text annotation (PU status + SNR)
annotationText = annotation('textbox',[0.15 0.8 0.3 0.1], ...
    'String','', 'FontSize',12, 'FontWeight','bold', ...
    'EdgeColor','none', 'Color','k');

%% Main Loop: Animation + Dataset Generation
for frame = 1:numFrames
    
    % If figure/lines were closed, stop cleanly
    if ~isgraphics(hTime) || ~isgraphics(hFFT)
        disp('Figure closed. Stopping animation loop.');
        break;
    end

    % ---- Randomly choose SNR and PU presence ----
    snr_dB = SNR_dB_list(randi(numel(SNR_dB_list)));  % random SNR from list
    pu_present = randi([0 1]);                        % 0 = absent, 1 = present
    
    % ---- Generate received signal y ----
    if pu_present == 1
        % BPSK PU signal
        bits    = randi([0 1], 1, N);
        symbols = 2*bits - 1;                 % BPSK: 0 -> -1, 1 -> +1
        s       = symbols;
        
        % Optional Rayleigh fading
        if includeRayleigh
            h = (randn(1,N) + 1j*randn(1,N))/sqrt(2);
            s = s .* h;
        end
        
        % AWGN for desired SNR
        signal_power = mean(abs(s).^2);
        snr_linear   = 10^(snr_dB/10);
        noise_power  = signal_power / snr_linear;
        n = sqrt(noise_power/2) * (randn(1,N) + 1j*randn(1,N));
        
        y = s + n;
    else
        % PU absent -> noise only
        noise_power = 1;
        y = sqrt(noise_power/2) * (randn(1,N) + 1j*randn(1,N));
    end
    
    % ---- Feature extraction for dataset ----
    E = sum(abs(y).^2) / N;   % normalized energy
    if row <= size(dataset,1)
        dataset(row, :) = [E, snr_dB, pu_present];
        row = row + 1;
    end
    
    % ---- Update animation ----
    % Time-domain (real part, limited samples)
    subplot(2,1,1);
    if isgraphics(hTime)
        set(hTime, 'XData', t(1:numTimeSamplesToShow), ...
                   'YData', real(y(1:numTimeSamplesToShow)));
    end
    if pu_present == 1
        title('Time Domain Signal (PU PRESENT)');
    else
        title('Time Domain Signal (PU ABSENT)');
    end
    
    % Spectrum
    subplot(2,1,2);
    Y = fftshift(fft(y, NFFT));
    magY_dB = 20*log10(abs(Y)+eps);
    if isgraphics(hFFT)
        set(hFFT, 'XData', f, 'YData', magY_dB);
    end
    title('Frequency Spectrum');
    
    % Update annotation text
    if pu_present == 1
        statusStr = sprintf('PU PRESENT  |  SNR = %d dB', snr_dB);
        annotationText.String = statusStr;
        annotationText.Color  = [1 0 0];  % red
    else
        statusStr = sprintf('PU ABSENT   |  SNR = %d dB', snr_dB);
        annotationText.String = statusStr;
        annotationText.Color  = [0 0.6 0];  % green
    end
    
    drawnow;
    pause(pauseTime);
end

%% Save dataset (for the rows actually filled)
if row > 1
    writematrix(dataset(1:row-1, :), 'spectrum_dataset_single.csv');
    fprintf('Saved dataset to spectrum_dataset_single.csv with %d samples\n', row-1);
else
    warning('No data collected, dataset not saved.');
end
