# Board-game-recommendation

## save_similarity.py

### defined function1: save_content_based_similarity
save_content_based_similarity에는 콘텐츠 기반 필터링이 구현되어 있습니다. 보드게임들 사이의 콘텐츠 기반 필터링을 진행하여 각 보드게임 사이의 코사인 유사도가 계산되어 있는 행렬을 'similarity.npy' 파일에 저장합니다.

### defined function2: save_corr_matrix
save_corr_matrix에는 아이템 기반 협업 필터링이 구현되어 있습니다. 보드게임들 사이의 아이템 기반 협업 필터링을 진행하여 각 보드게임 사이의 연관성이 계산되어 있는 행렬을 'corr_matrix.npy' 파일에 저장합니다.

## get_recommendation.py

### defined function1: get_content_based_reccomendation
get_content_based_reccomendation를 통해 특정 보드게임과 콘텐츠 기반 필터링으로 유사한 보드게임들을 확인할 수 있습니다. 매개변수로 어떤 보드게임에 대한 추천을 받을지, 몇개의 추천을 받을지를 전달받습니다.

### defined function2: get_colaborative_filtering_recommendation
get_colaborative_filtering_recommendation를 통해 특정 보드게임과 아이템 기반 협업 필터링으로 유사한 보드게임들을 확인할 수 있습니다. 매개변수로 어떤 보드게임에 대한 추천을 받을지, 몇개의 추천을 받을지를 전달받습니다.

## license
본 리포지토리에 사용된 데이터의 저작권은 모두 BoardGameGeeks에 있습니다. 자세한 사항은 LICENSE.md 파일을 참고하시기 바랍니다.
