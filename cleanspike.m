%outputsweeps = cleanspike (inputsweeps, stimulusindex, spikestartindex, spiketroughindex, spikeendindex, samplingfrequency)
%
%inputsweeps should be a matrix of sweeps-by-samples
%stimulusindex should be, in samples, the time of the main stimulation (set it to NaN if not applicable!, e.g. when no stimulation was delivered)
%spikestartindex should be, in samples, the time when the mean spike is considered to have begun (better to err on the side of including a bit more than needed)
%spiketroughindex should be, in samples, the time when the mean spike shape is at its lowest
%spikeendindex should be in, in samples, the time when the mean spike is considered to have ended (better to err on the side of including a bit more than needed)
%samplingfrequency should be the number of samples per second
%
%outputsweeps will be a matrix of sweeps-by-samples, with spike waveforms "cleaned" through "pattern-matched noise imputation"
%the optional second return value will be a matrix of sweeps-by-samples, with spike waveforms aligned but NOT "cleaned" through "pattern-matched noise imputation"

function [outputsweeps, varargout] = cleanspike (inputsweeps, stimulusindex, spikestartindex, spiketroughindex, spikeendindex, samplingfrequency)

    spikewindowms = 2.0; %ms, duration
    troughms = +0.3; %ms, relative to spike window start    
    alignmentsearchradiusms = 0.1; %ms, max for aligning troughs across sweeps

    prestimulusendms = -8; %ms, relative to stimulusindex, when the first of the two biphasic stimuli is definitely over (better to err on the side of making this interval bigger)
    mainstimulusstartms = -1; %ms, relative to stimulusindex, when the main stimulus is about to start soon (better to err on the side of making this interval bigger)
    mainstimulusendms = +2; %ms, relative to stimulusindex, when the main stimulus is definitely over (better to err on the side of making this interval bigger)

    alignmentsamples = ceil(alignmentsearchradiusms/1000*samplingfrequency); %in samples, search radius for trough alignment

    spikewindowstart = floor(spiketroughindex - troughms/1000*samplingfrequency);
    spikewindowend = ceil(spikewindowstart + spikewindowms/1000*samplingfrequency);
    spikewindown = numel(spikewindowstart:spikewindowend);
    analysiswindown = size(inputsweeps, 2);

    troughinwindow = 1+round(troughms/1000*samplingfrequency); %in samples, relative to spike window start
    
    currentlookfrom = max(spiketroughindex-alignmentsamples, 1);
    currentlookuntil = min(currentlookfrom+2*alignmentsamples, analysiswindown);

    for sweepi=1:size(inputsweeps, 1)
        [~, currentlocalminwhere] = min(inputsweeps(sweepi, currentlookfrom:currentlookuntil));
        displacementneeded = spiketroughindex - (currentlookfrom+currentlocalminwhere-1);
        if displacementneeded > 0
            inputsweeps(sweepi, 1+displacementneeded:end) = inputsweeps(sweepi, 1:end-displacementneeded);
            inputsweeps(sweepi, 1:displacementneeded) = NaN;
        elseif displacementneeded < 0
            inputsweeps(sweepi, 1:end+displacementneeded) = inputsweeps(sweepi, 1-displacementneeded:end);
            inputsweeps(sweepi, end+displacementneeded+1:end) = NaN;
        end
    end

    individualspikeshapes = inputsweeps(:, spikewindowstart:spikewindowend);
    originalindividualspikeshapes = individualspikeshapes;
    meanspikeshape = mean(individualspikeshapes, 1, 'omitmissing');
    originalmeanspikeshape = meanspikeshape;

    potentialnoisesweeps = inputsweeps;

    ispotentialnoise = true(1, analysiswindown);
    ispotentialnoise(spikestartindex:spikeendindex) = false; %exclude our spike from being considered as a potential source for imputation

    %exclude the prestimulus interval from being considered as a potential source for imputation
    prestimulusstartindex = 1;
    prestimulusendindex = min([ceil(stimulusindex + prestimulusendms/1000*samplingfrequency), analysiswindown], [], 'includenan'); %if it's NaN, it should remain NaN (to not exclude anything in that case)
    if ~isnan(prestimulusstartindex) && ~isnan(prestimulusendindex)
        ispotentialnoise(prestimulusstartindex:prestimulusendindex) = false;
    end

    %exclude the main stimulus interval from being considered as a potential source for imputation
    mainstimulusstartindex = max([floor(stimulusindex + mainstimulusstartms/1000*samplingfrequency), 1], [], 'includenan'); %if it's NaN, it should remain NaN (to not exclude anything in that case)
    mainstimulusendindex = min([ceil(stimulusindex + mainstimulusendms/1000*samplingfrequency), analysiswindown], [], 'includenan'); %if it's NaN, it should remain NaN (to not exclude anything in that case)
    if ~isnan(mainstimulusstartindex) && ~isnan(mainstimulusendindex)
        ispotentialnoise(mainstimulusstartindex:mainstimulusendindex) = false;
    end

    %exclude "probable spikes" from being considered as a potential source for imputation, now on a per-sweep basis
    for sweepi=1:size(individualspikeshapes, 1)
        currentpotentialnoise = potentialnoisesweeps(sweepi, :);
        currentpotentialnoise(~ispotentialnoise) = NaN; %to exclude stimulation artifacts from noise level estimatation
        currentspikethreshold1 = 3*median(abs(currentpotentialnoise), 'all', 'omitmissing')/0.6745; %per Quiroga RQ, Nadasdy Z, Ben-Shaul Y (2004) Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. Neural computation 16:1661–1687, which in turn references Donoho DL, Johnstone IM (1994) Ideal spatial adaptation by wavelet shrinkage. Biometrika 81:425–455.
        currentpotentialnoisestd = std(currentpotentialnoise, 'omitmissing');
        currentspikethreshold2 = 0 + 3*currentpotentialnoisestd; %the grand overall mean should be 0 due to highpass filtering
        currentspikethreshold = min([currentspikethreshold1, currentspikethreshold2]); %the more conservative of two thresholds (which will exclude more)
        potentialnoisesweeps(sweepi, ~ispotentialnoise | currentpotentialnoise > currentspikethreshold | currentpotentialnoise < -currentspikethreshold) = NaN;
    end

    % Processing BEFORE the trough
    dipbackbelowzeroindex = find([true(1, troughinwindow-3), false(1, spikewindown-troughinwindow+3)] & originalmeanspikeshape < 0 & [originalmeanspikeshape(1:end-1)-originalmeanspikeshape(2:end) < 0, false], 1, 'last'); %before the trough AND negative AND negative-direction
    if isempty(dipbackbelowzeroindex), dipbackbelowzeroindex = NaN; end
    needtoreplaceuntil = dipbackbelowzeroindex;
    needtoreplacen = dipbackbelowzeroindex;
    if needtoreplacen > 0
        for swi=1:size(individualspikeshapes, 1)
            %patterndifferences will contain, at each index (!), the weighted sum of squared distances (between the interval to be replaced and a potential noise interval) across a 3 datapoint "overlap" immediately AFTER the interval
            patterndifferences = [potentialnoisesweeps(swi, 2:end)-individualspikeshapes(swi, needtoreplaceuntil+1), NaN] .^ 2.0 * 1.0 + [potentialnoisesweeps(swi, 3:end)-individualspikeshapes(swi, needtoreplaceuntil+2), NaN, NaN] .^ 2.0 * 0.5 + [potentialnoisesweeps(swi, 4:end)-individualspikeshapes(swi, needtoreplaceuntil+3), NaN, NaN, NaN] .^ 2.0 * 0.25;
            %exclude intervals that would stretch outside of the analysis window or would include excluded timepoints
            for currentnoisei=1:analysiswindown
                if currentnoisei-needtoreplacen+1 < 1 || any(isnan(potentialnoisesweeps(swi, currentnoisei-needtoreplacen+1:currentnoisei)))
                    patterndifferences(currentnoisei) = NaN;
                end
            end
            [~, relativelybestwhere] = min(patterndifferences, [], 'omitmissing');
            individualspikeshapes(swi, 1:needtoreplaceuntil) = potentialnoisesweeps(swi, relativelybestwhere-needtoreplacen+1:relativelybestwhere);
        end
    end

    % Processing AFTER the trough
    returntozeroindex = find([false(1, troughinwindow+3), true(1, spikewindown-troughinwindow-3)] & originalmeanspikeshape < 0 & [false, originalmeanspikeshape(2:end)-originalmeanspikeshape(1:end-1) < 0], 1, 'first'); %after the trough AND negative AND negative-direction
    if isempty(returntozeroindex), returntozeroindex = NaN; end
    needtoreplacefrom = returntozeroindex;
    needtoreplacen = spikewindown-needtoreplacefrom+1;
    if needtoreplacen > 0
        for swi=1:size(individualspikeshapes, 1)
            %patterndifferences will contain, at each index (!), the weighted sum of squared distances (between the interval to be replaced and a potential noise interval) across a 3 datapoint "overlap" immediately BEFORE the interval
            patterndifferences = [NaN, potentialnoisesweeps(swi, 1:end-1)-individualspikeshapes(swi, needtoreplacefrom-1)] .^ 2.0 * 1.0 + [NaN, NaN, potentialnoisesweeps(swi, 1:end-2)-individualspikeshapes(swi, needtoreplacefrom-2)] .^ 2.0 * 0.5 + [NaN, NaN, NaN, potentialnoisesweeps(swi, 1:end-3)-individualspikeshapes(swi, needtoreplacefrom-3)] .^ 2.0 * 0.25;
            %exclude intervals that would stretch outside of the analysis window or would include excluded timepoints
            for currentnoisei=1:analysiswindown
                if currentnoisei+needtoreplacen-1 > analysiswindown || any(isnan(potentialnoisesweeps(swi, currentnoisei:currentnoisei+needtoreplacen-1)))
                    patterndifferences(currentnoisei) = NaN;
                end
            end
            [~, relativelybestwhere] = min(patterndifferences, [], 'omitmissing');
            individualspikeshapes(swi, needtoreplacefrom:end) = potentialnoisesweeps(swi, relativelybestwhere:relativelybestwhere+needtoreplacen-1);
        end
    end

    %final baseline compensation (because the median may have slightly changed as a result of the imputation, e.g. by removing a second big spike)
    meanspikeshape = mean(individualspikeshapes, 1, 'omitmissing');
    finalbaselinecompensation = median(meanspikeshape);
    outputsweeps = individualspikeshapes-finalbaselinecompensation;

    if nargout > 1
        varargout{1} = originalindividualspikeshapes;
    end

end