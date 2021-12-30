# Board-game-recommendation

## save_similarity.py

save_similarity 함수를 정의합니다. 해당 함수를 호출하면 데이터셋으로부터 보드게임별 이름, 최소 인원, 최대 인원, 카테고리, 진행방식에 대한 정보를 불러와 이를 바탕으로 컨텐츠 기반 필터링을 진행합니다. 컨텐츠 기반 필터링을 통해 생성된 유사도 행렬을 .npy 형식으로 save_similarity.py가 있는 디렉토리와 같은 디렉토리에 저장합니다.

## get_recommendation.py

get_k_recommendation 함수를 정의합니다. 해당 함수는 board_game_title과 k를 매개변수로 입력받습니다. 출력으로는 board_game_title로 입력된 이름의 보드게임과 유사한 k개의 보드게임을 리턴합니다.

## license
본 리포지토리에 사용된 데이터의 저작권은 모두 BoardGameGeeks에 있습니다. 자세한 사항은 LICENSE.md 파일을 참고하시기 바랍니다.
