window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
                const {
                    classes,
                    colorscale,
                    style,
                    colorProp
                } = context.hideout;
                const value = feature.properties[colorProp];
                let fillColor = colorscale[0];

                for (let i = 0; i < classes.length - 1; i++) {
                    if (value >= classes[i] && value < classes[i + 1]) {
                        fillColor = colorscale[i];
                        break;
                    }
                }

                if (value >= classes[classes.length - 1]) {
                    fillColor = colorscale[colorscale.length - 1];
                }

                return {
                    ...style,
                    fillColor
                };
            }

            ,
        function1: function(feature, context) {
            const {
                classes,
                colorscale,
                style,
                colorProp
            } = context.hideout;
            const value = feature.properties[colorProp];
            let fillColor = colorscale[0];
            for (let i = 0; i < classes.length; ++i) {
                if (value > classes[i]) fillColor = colorscale[i];
            }
            return {
                ...style,
                fillColor: fillColor
            };
        }

    }
});