clc;
clear;
close all;

%% ============================================================
%  OFDM over Time-Varying Rayleigh Fading Channel
%  Jakes Doppler Model + CP/No-CP Comparison
%  Multiple Independent Channel Realizations
% =============================================================

%% System Parameters
N = 128;                    % Number of OFDM subcarriers
cpLen = 10;                 % Cyclic prefix length
L = 10;                     % Number of channel paths
M = 4;                      % QPSK
numSymbols = 3000;          % OFDM symbols per realization

Fs = 1e6;                   % Sampling frequency (Hz)

SNR_dB = 0:2:20;

% Doppler frequencies used in the simulations
Doppler_values = [0 100 500 1000 2000];

% Doppler range for Plot 4
Doppler_range = [0 100 500 1000 2000 5000];

% Number of independent Rayleigh fading realizations
numRealizations = 20;

rng(1);

%% Channel Parameters
% Integer sample-spaced delays
pathDelays = (0:L-1)/Fs;

% Equal average power for all paths
avgPathGains = zeros(1,L);

%% ============================================================
% PLOT 1
% BER vs SNR with and without Cyclic Prefix
% =============================================================

fprintf('\n=============================================\n');
fprintf('PLOT 1: BER vs SNR with/without CP\n');
fprintf('=============================================\n');

BER_noCP = zeros(size(SNR_dB));
BER_CP   = zeros(size(SNR_dB));

% Use 500 Hz Doppler for the CP comparison
fD_plot1 = 500;

for snrIndex = 1:length(SNR_dB)

    totalErrors_CP = 0;
    totalBits_CP   = 0;

    totalErrors_noCP = 0;
    totalBits_noCP   = 0;

    for realization = 1:numRealizations

        %% Generate random QPSK data
        numBits = N * numSymbols * log2(M);

        txBits = randi([0 1], numBits, 1);

        % QPSK modulation
        txSymbols = pskmod( ...
            bi2de(reshape(txBits,2,[]).','left-msb'), ...
            M, pi/4, 'gray');

        X = reshape(txSymbols, N, numSymbols);

        %% OFDM modulation
        x = ifft(X,N,1) * sqrt(N);

        %% Add CP
        xCP = [x(end-cpLen+1:end,:); x];

        %% Serialize
        txCP = xCP(:);
        txNoCP = x(:);

        %% -----------------------------------------------------
        % Time-varying Rayleigh channel
        % ------------------------------------------------------

        % CP case
        rayChanCP = comm.RayleighChannel( ...
            'SampleRate', Fs, ...
            'PathDelays', pathDelays, ...
            'AveragePathGains', avgPathGains, ...
            'NormalizePathGains', true, ...
            'MaximumDopplerShift', fD_plot1, ...
            'DopplerSpectrum', doppler('Jakes'), ...
            'PathGainsOutputPort', true, ...
            'RandomStream', 'mt19937ar with seed', ...
            'Seed', 1000 + realization);

        [rxCP_channel, pathGainsCP] = rayChanCP(txCP);

        % No-CP case
        rayChanNoCP = comm.RayleighChannel( ...
            'SampleRate', Fs, ...
            'PathDelays', pathDelays, ...
            'AveragePathGains', avgPathGains, ...
            'NormalizePathGains', true, ...
            'MaximumDopplerShift', fD_plot1, ...
            'DopplerSpectrum', doppler('Jakes'), ...
            'PathGainsOutputPort', true, ...
            'RandomStream', 'mt19937ar with seed', ...
            'Seed', 2000 + realization);

        [rxNoCP_channel, pathGainsNoCP] = rayChanNoCP(txNoCP);

        %% Noise variance
        snrLinear = 10^(SNR_dB(snrIndex)/10);
        noiseVar = 1/snrLinear;

        %% Add AWGN
        noiseCP = sqrt(noiseVar/2) * ...
            (randn(size(rxCP_channel)) + ...
             1j*randn(size(rxCP_channel)));

        noiseNoCP = sqrt(noiseVar/2) * ...
            (randn(size(rxNoCP_channel)) + ...
             1j*randn(size(rxNoCP_channel)));

        rxCP = rxCP_channel + noiseCP;
        rxNoCP = rxNoCP_channel + noiseNoCP;

        %% =====================================================
        % CP CASE
        % ======================================================

        % Reshape into OFDM symbols
        rxCP = reshape(rxCP, N+cpLen, numSymbols);

        % Remove CP
        rxCP = rxCP(cpLen+1:end,:);

        % FFT
        Y_CP = fft(rxCP,N,1)/sqrt(N);

        %% Channel estimation
        H_CP = zeros(N,numSymbols);

        for s = 1:numSymbols

            centerIndex = (s-1)*(N+cpLen) + ...
                          cpLen + floor(N/2) + 1;

            h_inst = squeeze(pathGainsCP(centerIndex,:));

            H_CP(:,s) = fft(h_inst,N).';

        end

        %% Zero-Forcing equalization
        Hsafe = H_CP;

        threshold = 1e-8;
        Hsafe(abs(Hsafe) < threshold) = threshold;

        Z_CP = Y_CP ./ Hsafe;

        %% QPSK demodulation
        rxSymbols_CP = Z_CP(:);

        rxIndices_CP = pskdemod( ...
            rxSymbols_CP, M, pi/4, 'gray');

        rxBits_CP = reshape( ...
            de2bi(rxIndices_CP,2,'left-msb').', [], 1);

        %% Count errors
        totalErrors_CP = totalErrors_CP + ...
            sum(rxBits_CP ~= txBits);

        totalBits_CP = totalBits_CP + length(txBits);


        %% =====================================================
        % NO-CP CASE
        % ======================================================

        rxNoCP = reshape(rxNoCP, N, numSymbols);

        Y_noCP = fft(rxNoCP,N,1)/sqrt(N);

        %% Channel estimation
        H_noCP = zeros(N,numSymbols);

        for s = 1:numSymbols

            centerIndex = (s-1)*N + ...
                          floor(N/2) + 1;

            h_inst = squeeze(pathGainsNoCP(centerIndex,:));

            H_noCP(:,s) = fft(h_inst,N).';

        end

        %% Zero-Forcing equalization
        Hsafe = H_noCP;

        Hsafe(abs(Hsafe) < threshold) = threshold;

        Z_noCP = Y_noCP ./ Hsafe;

        %% QPSK demodulation
        rxSymbols_noCP = Z_noCP(:);

        rxIndices_noCP = pskdemod( ...
            rxSymbols_noCP, M, pi/4, 'gray');

        rxBits_noCP = reshape( ...
            de2bi(rxIndices_noCP,2,'left-msb').', [], 1);

        %% Count errors
        totalErrors_noCP = totalErrors_noCP + ...
            sum(rxBits_noCP ~= txBits);

        totalBits_noCP = totalBits_noCP + length(txBits);

    end

    %% Average BER
    BER_CP(snrIndex) = totalErrors_CP / totalBits_CP;
    BER_noCP(snrIndex) = totalErrors_noCP / totalBits_noCP;

    fprintf('SNR = %2d dB : No CP = %.5e, With CP = %.5e\n', ...
        SNR_dB(snrIndex), ...
        BER_noCP(snrIndex), ...
        BER_CP(snrIndex));

end


%% Plot 1
figure;

semilogy(SNR_dB, BER_noCP, 'r-o', ...
    'LineWidth',2, 'MarkerSize',8);

hold on;

semilogy(SNR_dB, BER_CP, 'b-s', ...
    'LineWidth',2, 'MarkerSize',8);

grid on;

xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');

title('BER vs SNR for OFDM over Time-Varying Rayleigh Channel');

legend('Without Cyclic Prefix', ...
       'With Cyclic Prefix', ...
       'Location','southwest');

set(gca,'FontSize',12);


%% ============================================================
% PLOTS 2, 3 AND 4
% Use one common simulation function
% =============================================================

fprintf('\n=============================================\n');
fprintf('Running Doppler simulations...\n');
fprintf('=============================================\n');

% Calculate normalized Doppler
T_OFDM = N/Fs;

fprintf('\nOFDM useful symbol duration = %.3f microseconds\n', ...
    T_OFDM*1e6);

for fD = Doppler_range
    fprintf('fD = %5d Hz --> fD*T_OFDM = %.4f\n', ...
        fD, fD*T_OFDM);
end


%% ============================================================
% PLOT 2
% BER vs Doppler at SNR = 10 dB
% =============================================================

SNR_fixed = 10;

BER_vs_Doppler = zeros(size(Doppler_range));

fprintf('\n=============================================\n');
fprintf('PLOT 2: BER vs Doppler at SNR = 10 dB\n');
fprintf('=============================================\n');

for d = 1:length(Doppler_range)

    fD = Doppler_range(d);

    BER_temp = runOFDMsimulation( ...
        N, cpLen, L, M, Fs, numSymbols, ...
        fD, SNR_fixed, numRealizations);

    BER_vs_Doppler(d) = BER_temp;

    fprintf('fD = %5d Hz : BER = %.5e\n', ...
        fD, BER_temp);

end


%% Plot 2
figure;

plot(Doppler_range, BER_vs_Doppler, 'k-o', ...
    'LineWidth',2, 'MarkerSize',8);

grid on;

xlabel('Maximum Doppler Frequency (Hz)');
ylabel('Bit Error Rate (BER)');

title('BER vs Doppler Frequency at SNR = 10 dB');

set(gca,'FontSize',12);


%% ============================================================
% PLOT 3
% BER vs SNR for different Doppler frequencies
% =============================================================

BER_Doppler_SNR = zeros(length(Doppler_values), ...
                         length(SNR_dB));

fprintf('\n=============================================\n');
fprintf('PLOT 3: BER vs SNR for Different Dopplers\n');
fprintf('=============================================\n');

for d = 1:length(Doppler_values)

    fD = Doppler_values(d);

    fprintf('\nDoppler = %d Hz\n',fD);

    for s = 1:length(SNR_dB)

        BER_Doppler_SNR(d,s) = runOFDMsimulation( ...
            N, cpLen, L, M, Fs, numSymbols, ...
            fD, SNR_dB(s), numRealizations);

        fprintf('SNR = %2d dB : BER = %.5e\n', ...
            SNR_dB(s), BER_Doppler_SNR(d,s));

    end

end


%% Plot 3
figure;

markers = {'o','s','^','d','*'};

for d = 1:length(Doppler_values)

    semilogy(SNR_dB, ...
        BER_Doppler_SNR(d,:), ...
        ['-' markers{d}], ...
        'LineWidth',2, ...
        'MarkerSize',8);

    hold on;

end

grid on;

xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');

title('BER vs SNR for Different Doppler Frequencies');

legend( ...
    'f_D = 0 Hz', ...
    'f_D = 100 Hz', ...
    'f_D = 500 Hz', ...
    'f_D = 1000 Hz', ...
    'f_D = 2000 Hz', ...
    'Location','southwest');

set(gca,'FontSize',12);


%% ============================================================
% PLOT 4
% BER vs Doppler for Different SNRs
% =============================================================

SNR_values_plot4 = [0 5 10 15 20];

BER_Doppler_SNR2 = zeros(length(SNR_values_plot4), ...
                          length(Doppler_range));

fprintf('\n=============================================\n');
fprintf('PLOT 4: BER vs Doppler for Different SNRs\n');
fprintf('=============================================\n');

for s = 1:length(SNR_values_plot4)

    snrValue = SNR_values_plot4(s);

    fprintf('\nSNR = %d dB\n',snrValue);

    for d = 1:length(Doppler_range)

        fD = Doppler_range(d);

        BER_Doppler_SNR2(s,d) = runOFDMsimulation( ...
            N, cpLen, L, M, Fs, numSymbols, ...
            fD, snrValue, numRealizations);

        fprintf('fD = %5d Hz : BER = %.5e\n', ...
            fD, BER_Doppler_SNR2(s,d));

    end

end


%% Plot 4
figure;

markers2 = {'o','s','^','d','*'};

for s = 1:length(SNR_values_plot4)

    semilogy(Doppler_range, ...
        BER_Doppler_SNR2(s,:), ...
        ['-' markers2{s}], ...
        'LineWidth',2, ...
        'MarkerSize',8);

    hold on;

end

grid on;

xlabel('Maximum Doppler Frequency (Hz)');
ylabel('Bit Error Rate (BER)');

title('BER vs Doppler Frequency for Different SNRs');

legend( ...
    'SNR = 0 dB', ...
    'SNR = 5 dB', ...
    'SNR = 10 dB', ...
    'SNR = 15 dB', ...
    'SNR = 20 dB', ...
    'Location','southeast');

set(gca,'FontSize',12);


%% ============================================================
% PLOT 5
% Time-Varying Rayleigh Channel Magnitude
% =============================================================

fD_plot = 2000;

numPlotSymbols = 80;

% Flat Rayleigh channel for displaying h(t)
rayChanPlot = comm.RayleighChannel( ...
    'SampleRate', Fs, ...
    'PathDelays', 0, ...
    'AveragePathGains', 0, ...
    'NormalizePathGains', true, ...
    'MaximumDopplerShift', fD_plot, ...
    'DopplerSpectrum', doppler('Jakes'), ...
    'PathGainsOutputPort', true, ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', 9999);

dummySignal = ones(numPlotSymbols*N,1);

[~, pathGainsPlot] = rayChanPlot(dummySignal);

h_t = pathGainsPlot(:,1);

time_ms = (0:length(h_t)-1)/Fs*1000;


%% Plot 5
figure;

plot(time_ms, abs(h_t), ...
    'LineWidth',1.5);

grid on;

xlabel('Time (ms)');
ylabel('|h(t)|');

title(sprintf( ...
    'Time-Varying Rayleigh Channel Magnitude, f_D = %d Hz', ...
    fD_plot));

set(gca,'FontSize',12);


%% ============================================================
% FINAL SUMMARY
% =============================================================

fprintf('\n=============================================\n');
fprintf('SIMULATION COMPLETED\n');
fprintf('=============================================\n');

fprintf('Number of subcarriers       : %d\n',N);
fprintf('CP length                   : %d\n',cpLen);
fprintf('Number of channel paths     : %d\n',L);
fprintf('QPSK modulation             : %d-QAM equivalent\n',M);
fprintf('OFDM symbols/realization    : %d\n',numSymbols);
fprintf('Independent realizations    : %d\n',numRealizations);
fprintf('Sampling frequency          : %.2f MHz\n',Fs/1e6);
fprintf('=============================================\n');


%% ============================================================
% LOCAL FUNCTION
% =============================================================

function BER = runOFDMsimulation( ...
    N, cpLen, L, M, Fs, numSymbols, ...
    fD, SNR_dB, numRealizations)

    %% Channel parameters
    pathDelays = (0:L-1)/Fs;
    avgPathGains = zeros(1,L);

    totalErrors = 0;
    totalBits = 0;

    %% ========================================================
    % Independent channel realizations
    % ========================================================

    for realization = 1:numRealizations

        %% ----------------------------------------------------
        % Generate random QPSK data
        % -----------------------------------------------------

        numBits = N * numSymbols * log2(M);

        txBits = randi([0 1], numBits, 1);

        txIndices = bi2de( ...
            reshape(txBits,2,[]).', ...
            'left-msb');

        txSymbols = pskmod( ...
            txIndices, M, pi/4, 'gray');

        X = reshape(txSymbols,N,numSymbols);

        %% ----------------------------------------------------
        % OFDM modulation
        % -----------------------------------------------------

        x = ifft(X,N,1)*sqrt(N);

        %% Add cyclic prefix
        xCP = [x(end-cpLen+1:end,:); x];

        %% Serialize
        tx = xCP(:);

        %% ----------------------------------------------------
        % Create independent Jakes Rayleigh channel
        % -----------------------------------------------------

        rayChan = comm.RayleighChannel( ...
            'SampleRate', Fs, ...
            'PathDelays', pathDelays, ...
            'AveragePathGains', avgPathGains, ...
            'NormalizePathGains', true, ...
            'MaximumDopplerShift', fD, ...
            'DopplerSpectrum', doppler('Jakes'), ...
            'PathGainsOutputPort', true, ...
            'RandomStream', 'mt19937ar with seed', ...
            'Seed', 5000 + realization);

        %% Pass signal through channel
        [rx_channel,pathGains] = rayChan(tx);

        %% ----------------------------------------------------
        % AWGN
        % -----------------------------------------------------

        snrLinear = 10^(SNR_dB/10);

        noiseVar = 1/snrLinear;

        noise = sqrt(noiseVar/2) * ...
            (randn(size(rx_channel)) + ...
             1j*randn(size(rx_channel)));

        rx = rx_channel + noise;

        %% ----------------------------------------------------
        % Reshape OFDM symbols
        % -----------------------------------------------------

        rx = reshape(rx,N+cpLen,numSymbols);

        %% Remove CP
        rx = rx(cpLen+1:end,:);

        %% FFT
        Y = fft(rx,N,1)/sqrt(N);

        %% ----------------------------------------------------
        % Channel frequency response
        % -----------------------------------------------------

        H = zeros(N,numSymbols);

        for s = 1:numSymbols

            % Middle of the useful OFDM symbol
            centerIndex = ...
                (s-1)*(N+cpLen) + ...
                cpLen + floor(N/2) + 1;

            % Instantaneous multipath channel
            h_inst = squeeze(pathGains(centerIndex,:));

            % Frequency response
            H(:,s) = fft(h_inst,N).';

        end

        %% ----------------------------------------------------
        % Zero-Forcing Equalization
        % -----------------------------------------------------

        Hsafe = H;

        threshold = 1e-8;

        Hsafe(abs(Hsafe)<threshold) = threshold;

        Z = Y ./ Hsafe;

        %% ----------------------------------------------------
        % QPSK Demodulation
        % -----------------------------------------------------

        rxSymbols = Z(:);

        rxIndices = pskdemod( ...
            rxSymbols,M,pi/4,'gray');

        rxBits = reshape( ...
            de2bi(rxIndices,2,'left-msb').', ...
            [],1);

        %% ----------------------------------------------------
        % BER calculation
        % -----------------------------------------------------

        totalErrors = totalErrors + ...
            sum(rxBits ~= txBits);

        totalBits = totalBits + length(txBits);

    end

    %% Average BER over all independent realizations
    BER = totalErrors/totalBits;

end