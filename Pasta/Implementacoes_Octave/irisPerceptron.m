%
% Gradient descent algo for linear regression
clear all; close all; clc; clf;
rawdata = load("iris2.dat", "ascii");

% set constants
scaled = true;
 
% scale and set the data
Y = rawdata(:,5)';
if scaled
        scaledX1 = (rawdata(:,3)' - min(rawdata(:,3)') ) ./ range(rawdata(:,3));
        scaledX2 = (rawdata(:,4)' - min(rawdata(:,4)'))./ range(rawdata(:,4));
        X = [ ones(1, rows(rawdata)); scaledX1; scaledX2];
        xaxis = [0 1]; yaxis = [0 1];
else
        X = [ ones(1, rows(rawdata)); rawdata(:,3)'; rawdata(:,4)'];
        xaxis = [min(X(2,:)) max(X(2,:))]; yaxis = [min(X(3,:)) max(X(3,:))];
endif

% gathering the class points
c1x = X(2, find(Y == 1));
c1y = X(3, find(Y == 1));

c2x = X(2, find(Y == -1));
c2y = X(3, find(Y == -1));

figure(1); 
plot(c1x, c1y, 'ro'); % plot the class 1
hold on;
plot(c2x, c2y, 'bo'); % plot the class 2
hold on;

disp("Press to start the learning...")


MAX_ITER =  300;          % maximum number of iterations

%err = zeros(1,MAX_ITER);  % initialize the error vector
%figure(2); plot(err,"g");

pause()  % wait for a key press

% GRADIENT DESCENT
m = columns(X); % number of instances
n = rows(X);    % number of parameters
W = zeros(n, 1);     % initialize W to all zeros
alpha = 0.4;    % gradient descent step size
 
disp('Starting Weights:');
W

% do the loop for W estimation using gradient descent
for iter = 1 : MAX_ITER

        Wc = W;  % save the current weights
        %err(iter) = mean(((Wc'*X) - Y).^2);  % calculate the error function E(w)
        %if (rem(iter,10) == 0) 
        %   figure(2); plot(err, "g");   
        %endif           

        for i = 1 : m
                
                yd = Wc' * X;
                yd(find(yd > 0)) = 1; yd(find(yd <= 0)) = -1;
                
                % looping for each weight W(i)
                for j = 1 : n       
                    % calculate new value of W(i)
                    W(j) = W(j) + alpha * (Y(i) - yd(i)) * X(j,i);
                end
        end
        
        xx = - W(1) / W(3);
        xy = - W(1) / W(2);
        fr1 = (0.8 / MAX_ITER);
        gColor = (fr1 * iter) + 0.1;
        axis([xaxis yaxis]); 
        
        if (rem(iter,10) == 0) 
            figure(1); line([0 xy], [xx 0], "Color", [ 0 gColor 0]);
        endif
        
        %fr1 = (0.8 / MAX_ITER);
        %gColor = (fr1 * iter) + 0.1;
        %figure(1); plot(X(2,:),newY, 'Color', [0 gColor 0]);
        %hold on
        if(W == Wc) break endif;

        pause(0.001); 
end

W

%err

% finding the decision boundary
xx = - W(1) / W(3);
xy = - W(1) / W(2);

% plotting the decision boundary along with the class points
figure(3);
plot(c1x, c1y, 'ro'); % plot the class 1
hold on;
plot(c2x, c2y, 'bo'); % plot the class 2
hold on;
axis([xaxis yaxis]); 
line([0 xy], [xx 0], "linewidth", 2, "Color", [ 0.03 0.54 0]);

hold off
