% Gradient descent algorithm for linear regression
clear all; close all; clc; clf;
rawdata = load("cirrose.dat", "ascii");
 
%set the data
% Selects the fifth attribute as input and scale the input data
scaledX = (rawdata(:,5)' - mean(rawdata(:,5)) ) ./ range(rawdata(:,5)');
% Loads the input values into X with a column of ones 
% and the values of X (whether scaled or not, uncomment to switch)
X = [ ones(1,rows(rawdata)); scaledX]; %rawdata(:,5)']; 
% Loads the output values into Y
Y = rawdata(:,7)';

% Solving by normal equation we can get the optimal values of W's.
% **Note that here the normal equation is slightly different from its definition,
% **it's because the instances are placed as columns rather than rows.
% Solve the normal equation and set the actual values of W's
W = pinv(X * X') * X * Y';
% Compute the outputs Y's, for the model with optimal values of W's.
% It'll be useful as reference line when plotting
YAct = (W' * X);

hold on;
% Plots the training data
figure(1); plot(X(2,:),Y, 'x');

MAX_ITER =  150;          % maximum number of iterations
err = zeros(1,MAX_ITER);  % initialize the error vector

% Initialize the second plot window for the error
figure(2); plot(err);

disp("Press to start the learning...")
pause()  % wait for a key press

% GRADIENT DESCENT
m = columns(X); % number of instances
n = rows(X);    % number of parameters
W = zeros(n,1); % initialize W to all zeros
alpha = 0.4;    % gradient descent step size

% Show the start weights 
disp('Starting Weights:');
W

% do the loop for W estimation using gradient descent
for iter = 1 : MAX_ITER

        Wc = W;  % save the current weights
        err(iter) = mean(((Wc'*X) - Y).^2);  % calculate the error function E(w)
        %err(iter) = mean( ((Wc'*X) - Y) .* X(i,:) );
        if (rem(iter,10) == 0)
          figure(2); plot(err);                % and plot it
        endif

        % looping for each weight W(i)
        for i = 1 : n       
            % calculate new value of W(i)
            W(i) = W(i) - alpha * mean( ((Wc'*X) - Y) .* X(i,:) );
        end
        % calculate the outputs from the new hypothesis
        newY = (W' * X); 
        
        fr1 = (0.8 / MAX_ITER);
        gColor = (fr1 * iter) + 0.1; % calculate the value of the color for plot
        
        if (rem(iter,10) == 0) 
          figure(1); plot(X(2,:),newY, 'Color', [0 gColor 0]);
        endif  
        hold on
        
        % *optional - slow down the execution for better visualization
        pause(0.00001); 
end


% Show the final weights
disp ('Final calculated weights');
W
% Plots the training data in a new window
figure(3); plot(X(2,:),Y, 'x'); hold on;
newY = (W' * X); % calculate the outputs from the new hypothesis
% Plots the found hypothesis
figure(3); plot(X(2,:),newY, 'Color', [0 1 0], "linewidth", 2); hold on;
% Plots the reference hypothesis
figure(3); plot(X(2,:),YAct, 'Color',[1 0 0], "linewidth", 2);
% finish off
hold off;
